import os
import sys
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.utils.data as utils_data
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn import metrics
import torch.nn.functional as F
import math
from time import time
import pandas as pd

import kpis
import export

font = {'size': 22}
matplotlib.rcParams['figure.figsize'] = (18, 12)
matplotlib.rc('font', **font)


# =============================================================================
# Internal Logging Helpers  (train-lscstm.py 스타일)
# =============================================================================

def _fmt_time(seconds):
    """초를 HH:MM:SS 문자열로 변환."""
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _print_train_progress(itr, total_itrs, metrics_dict, lr, elapsed, eta):
    """
    itr%100==0 마다 두 줄로 학습 진행 상황 출력.
      Line 1: 진행률 / elapsed / ETA
      Line 2: total loss  |  partial losses (나머지 항목)
    metrics_dict에서 'loss' 키(대소문자 무관)를 total loss로, 나머지를 partial로 분류.
    """
    scalars = {k: v for k, v in metrics_dict.items() if 'histogram' not in k}

    # total loss 키 탐색 (정확히 'loss' 이거나 'total'이 포함된 것 우선)
    total_key = None
    for k in scalars:
        if k.lower() == 'loss' or 'total' in k.lower():
            total_key = k
            break
    if total_key is None and scalars:
        total_key = next(iter(scalars))  # fallback: 첫 번째 키

    partial = {k: v for k, v in scalars.items() if k != total_key}

    progress  = itr / total_itrs * 100
    total_str = f"{total_key}: {scalars[total_key]:.4f}" if total_key else ""
    part_str  = "  ".join(f"{k}: {v:.4f}" for k, v in partial.items())

    line1 = (
        f"[Train] {itr:>6}/{total_itrs}  ({progress:5.1f}%)  |  "
        f"Elapsed: {_fmt_time(elapsed)}  ETA: {_fmt_time(eta)}"
    )
    line2 = f"        {total_str}"
    if part_str:
        line2 += f"  |  {part_str}"

    print(f"\033[F\033[K{line1}\n\033[K", end='\r')
    sys.stdout.flush()


def _print_val_summary(val_print_dict, val_kpi_dict, best_itr, best_val_score, val_score_key):
    """
    val 지표를 4줄로 분리 출력:
      [Loss   ] Total / Traj / Man Loss 등 주요 loss
      [Partial] *Partial* 키 계열
      [KPI    ] kpi_dict scalar 값들
      [Best   ] best itr / score / val_time
    """
    _SKIP = ('histogram', 'list', 'group', 'Best ITR', 'Best Val Score',
             'Validation Time', 'Itr')

    # ── Loss 계열 / Partial 계열 분리 ────────────────────────────────
    loss_items    = {}
    partial_items = {}
    for k, v in val_print_dict.items():
        if any(s in k for s in _SKIP) or 'histogram' in k:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if 'Partial' in k:
            partial_items[k] = fv
        else:
            loss_items[k] = fv

    if loss_items:
        print("[Loss   ] " + "  ".join(f"{k}: {v:.4f}" for k, v in loss_items.items()))
    if partial_items:
        print("[Partial] " + "  ".join(f"{k}: {v:.4f}" for k, v in partial_items.items()))

    # ── KPI 블록 ─────────────────────────────────────────────────────
    kpi_items = {
        k: val_kpi_dict[k] for k in val_kpi_dict
        if all(x not in k for x in ('histogram', 'list', 'group', 'min', 'max', 'mnll', 'rmse_table'))
        and not isinstance(val_kpi_dict[k], np.ndarray)
    }
    if kpi_items:
        print("[KPI    ] " + "  ".join(f"{k}: {v:.4f}" for k, v in kpi_items.items()))

    # ── Best / timing ─────────────────────────────────────────────────
    val_time = val_print_dict.get('Validation Time', 0)
    print(
        f"[Best   ] itr={best_itr}  |  "
        f"{val_score_key}={best_val_score:.4f}  |  "
        f"val_time={val_time:.1f}s"
    )
    print("\n")


def _print_best_ckpt(itr, best_val_score, save_path):
    print(f"[CKPT ] ✅  Best updated @ itr {itr}  (score={best_val_score:.4f})  →  {save_path}")


def _print_val_batch_progress(batch_idx, n_batches):
    progress = (batch_idx + 1) / n_batches * 100
    msg = f"Validating... {batch_idx+1}/{n_batches} ({progress:.1f}%)"
    print(msg + "   " * 5, end='\r')
    sys.stdout.flush()


# =============================================================================
# Public API
# =============================================================================

def deploy_top_func(p, model_deploy_func, model, de_dataset, device):
    model = model.to(device)
    de_loader = utils_data.DataLoader(
        dataset=de_dataset, shuffle=False,
        batch_size=p.BATCH_SIZE, drop_last=False, num_workers=16, pin_memory=True,
    )
    vis_data_path  = p.VIS_DIR   + p.experiment_tag + '.pickle'
    best_model_path = p.WEIGHTS_DIR + p.experiment_tag + '.pt'
    figure_name     = p.experiment_tag

    if p.SELECTED_MODEL != 'CONSTANT_PARAMETER':
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    export_dict = deploy_model(
        p, model, model_deploy_func, de_loader, de_dataset,
        device, vis_data_path=vis_data_path, figure_name=figure_name,
    )
    return export_dict


def eval_top_func(p, model_eval_func, model_kpi_func, model, loss_func_tuple,
                  te_dataset, device, tensorboard=None):
    model = model.to(device)
    te_loader = utils_data.DataLoader(
        dataset=te_dataset, shuffle=True,
        batch_size=p.BATCH_SIZE, drop_last=True, pin_memory=True, num_workers=16,
    )
    vis_data_path   = p.VIS_DIR    + p.experiment_tag + '.pickle'
    best_model_path = p.WEIGHTS_DIR + 'best.pt'
    figure_name     = p.experiment_tag

    if p.SELECTED_MODEL != 'CONSTANT_PARAMETER':
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    print_dict, kpi_dict = eval_model(
        p, tensorboard, model_eval_func, model_kpi_func,
        model, loss_func_tuple, te_loader, te_dataset,
        ' N/A', device, eval_type='Test',
        vis_data_path=vis_data_path, figure_name=figure_name,
    )

    # ── Test 결과 출력 ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  TEST KPIs")
    print(f"{'='*65}")
    scalar_pd = [(k, v) for k, v in print_dict.items() if 'histogram' not in k]
    for k, v in scalar_pd:
        print(f"  {k:<35} {v}")
    for k in kpi_dict:
        if 'histogram' not in k:
            print(f"  {k:<35} {kpi_dict[k]}")
    print(f"{'='*65}\n")

    return kpi_dict


def train_top_func(p, model_train_func, model_eval_func, model_kpi_func,
                   model, loss_func_tuple, optimizer,
                   tr_dataset, val_dataset, device,
                   prev_best_model=None, prev_itr=1, tensorboard=None,
                   log_callbacks=None):

    tr_loader = utils_data.DataLoader(
        dataset=tr_dataset, shuffle=True,
        batch_size=p.BATCH_SIZE, drop_last=True,
        num_workers=16, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,  
    )
    val_loader = utils_data.DataLoader(
        dataset=val_dataset, shuffle=True,
        batch_size=p.BATCH_SIZE, drop_last=True,
        num_workers=16, pin_memory=True,             
        persistent_workers=True, prefetch_factor=4, 
    )

    best_model_path = p.WEIGHTS_DIR + 'best.pt'
    if prev_best_model is not None:
        model.load_state_dict(torch.load(prev_best_model, map_location=device))

    best_val_score = float("inf") if p.LOWER_BETTER_VAL_SCORE else 0
    best_itr = 0
    p.LR_WU_CURRENT_BATCH = 0

    window_loss_sum  = {}   
    window_time_sum  = 0.0
    window_start     = time()
    training_start   = time()

    tr_iterator = iter(tr_loader)

    for itr in range(prev_itr, p.NUM_ITRS):
        itr_start = time()

        try:
            batch_data = next(tr_iterator)
        except StopIteration:
            tr_iterator = iter(tr_loader)
            batch_data = next(tr_iterator)

        # 4. train_step 호출 (tr_loader 대신 batch_data 전달)
        tr_print_dict, lr = train_step(
            p, model_train_func, model,
            loss_func_tuple, optimizer,
            tr_dataset, batch_data, itr, device,
        )

        itr_time = time() - itr_start

        # ── 100 itr 누적 ─────────────────────────────────────────────
        for k, v in tr_print_dict.items():
            if 'histogram' not in k:
                window_loss_sum[k] = window_loss_sum.get(k, 0.0) + v
        window_time_sum += itr_time

        # ── 100 itr마다 한 줄 출력 (\r) ──────────────────────────────
        if itr % 100 == 99:
            avg_metrics = {k: window_loss_sum[k] / 100 for k in window_loss_sum}
            elapsed = time() - training_start
            avg_itr_time = window_time_sum / 100
            eta = avg_itr_time * (p.NUM_ITRS - itr)
            _print_train_progress(itr + 1, p.NUM_ITRS, avg_metrics, lr, elapsed, eta)
            # 초기화
            window_loss_sum = {}
            window_time_sum = 0.0

        # ── TensorBoard (train) ───────────────────────────────────────
        ''''''
        if itr % 100 == 99:
            tensorboard.add_scalar('LR', lr, itr)
            for k in tr_print_dict:
                if 'histogram' in k:
                    tensorboard.add_histogram('Train_itr_' + k, tr_print_dict[k], itr)
                else:
                    tensorboard.add_scalar('Train_itr_' + k, tr_print_dict[k], itr)
        ''''''
        # ── Validation ───────────────────────────────────────────────
        if itr % p.VAL_FREQ == 0 or p.DEBUG_MODE:

            val_start = time()
            val_print_dict, val_kpi_dict = eval_model(
                p, tensorboard,
                model_eval_func, model_kpi_func,
                model, loss_func_tuple,
                val_loader, val_dataset,
                itr, device,
                eval_type='Validation',
            )
            val_elapsed = time() - val_start

            # val score 추출
            val_score = (val_print_dict[p.VAL_SCORE]
                         if p.VAL_SCORE in val_print_dict
                         else val_kpi_dict[p.VAL_SCORE])

            # best 갱신
            is_best = (p.LOWER_BETTER_VAL_SCORE and val_score < best_val_score) \
                   or (not p.LOWER_BETTER_VAL_SCORE and val_score > best_val_score)
            if is_best:
                best_val_score = val_score
                best_itr = itr
                torch.save(model.state_dict(), best_model_path)
                _print_best_ckpt(itr, best_val_score, best_model_path)

            val_print_dict['Validation Time']                       = val_elapsed
            val_print_dict['Best ITR']                              = best_itr
            val_print_dict[f'Best Val Score ({p.VAL_SCORE})']       = best_val_score

            _print_val_summary(val_print_dict, val_kpi_dict, best_itr,
                                best_val_score, p.VAL_SCORE)

            total_elapsed = time() - training_start
            h, rem = divmod(int(total_elapsed), 3600)
            m, s   = divmod(rem, 60)
            print()
            '''
            # TensorBoard (val)
            for k in val_print_dict:
                try:
                    tensorboard.add_scalar('Validation_itr_' + k, val_print_dict[k], itr)
                except Exception:
                    pass
            for k in val_kpi_dict:
                if 'histogram' in k:
                    tensorboard.add_histogram('Validation_' + k, val_kpi_dict[k], itr)
                elif 'rmse_table' in k:
                    for i in range(p.MAX_IN_SEQ_LEN):
                        tensorboard.add_scalar('Validation_' + k + str(i), val_kpi_dict[k][0, i], itr)
                elif all(x not in k for x in ('group', 'min', 'max', 'mnll', 'list')):
                    try:
                        tensorboard.add_scalar('Validation_' + k, val_kpi_dict[k], itr)
                    except Exception:
                        pass
            '''
            if p.DEBUG_MODE:
                print("\n[DEBUG] Debug mode — stopping after first validation.")
                break

    print(f"\n{'='*65}")
    print(f"  Training complete.  Best itr: {best_itr}  |  Best score: {best_val_score:.4f}")
    print(f"{'='*65}\n")

    return {
        'Best Itr': best_itr,
        'Best Validation Loss': best_val_score,
    }


def train_step(p, model_train_func, model, loss_func_tuple,
               optimizer, train_dataset, batch_data,
               itr, device):
    model.train()

    (data_tuple, man, _) = batch_data

    data_tuple = [data.to(device) for data in data_tuple]
    man = man.to(device)

    optimizer.zero_grad()

    loss, batch_print_info_dict = model_train_func(
        p, data_tuple, man, model, train_dataset, loss_func_tuple, device,
    )
    loss.backward()

    # LR 스케줄
    if p.LR_WU and itr <= p.LR_WU_BATCHES:
        lr = (p.LR * itr) / p.LR_WU_BATCHES / math.sqrt(p.LR_WU_BATCHES)
    elif p.LR_DECAY == 'inv-sqrt':
        lr = p.LR / math.sqrt(max(itr, 1))
    elif p.LR_DECAY == 'none':
        lr = p.LR
    else:
        raise ValueError(f'Unknown LR decay type: {p.LR_DECAY}')

    for g in optimizer.param_groups:
        g['lr'] = lr

    optimizer.step()

    return batch_print_info_dict, lr


def deploy_model(p, model, model_deploy_func, de_loader, de_dataset,
                 device, vis_data_path=None, figure_name=None):
    export_dict = {}
    n = len(de_loader)

    for batch_idx, (data_tuple, man, plot_info) in enumerate(de_loader):
        if p.DEBUG_MODE and batch_idx > 2:
            break

        data_tuple = [data.to(device) for data in data_tuple]
        with torch.no_grad():
            batch_export_dict = model_deploy_func(
                p, data_tuple, plot_info, de_dataset, model, device,
            )

        if batch_idx % 100 == 0:
            print(f"[Deploy] {batch_idx}/{n} ({batch_idx/n*100:.1f}%)", end='\r')
            sys.stdout.flush()

        if batch_idx == 0:
            for k in batch_export_dict:
                export_dict[k] = [batch_export_dict[k]]
        else:
            for k in batch_export_dict:
                export_dict[k].append(batch_export_dict[k])

    print()
    return export_dict


def eval_model(p, tb, model_eval_func, model_kpi_func, model, loss_func_tuple,
               test_loader, test_dataset, epoch, device,
               eval_type='Validation', vis_data_path=None, figure_name=None):

    print_dict     = {}
    kpi_input_dict = {}
    n_batches = len(test_loader) if eval_type != 'Validation' else p.MAX_VAL_ITR

    model.eval()

    with torch.no_grad():
        for batch_idx, (data_tuple, man, plot_info) in enumerate(test_loader):
            if eval_type == 'Validation' and batch_idx >= p.MAX_VAL_ITR:
                break
            if p.DEBUG_MODE and batch_idx > 2:
                break

            data_tuple = [data.to(device, non_blocking=True) for data in data_tuple]
            man = man.to(device, non_blocking=True)

            batch_print_info_dict, batch_kpi_input_dict = model_eval_func(
                p, data_tuple, man, plot_info,
                test_dataset, model, loss_func_tuple, device, eval_type,
            )

            for k in batch_print_info_dict:
                if 'histogram' not in k:
                    print_dict[k] = print_dict.get(k, 0.0) + batch_print_info_dict[k] / n_batches

            for k in batch_kpi_input_dict:
                if k not in kpi_input_dict:
                    kpi_input_dict[k] = []
                kpi_input_dict[k].append(batch_kpi_input_dict[k])

            if batch_idx % 10 == 0:
                _print_val_batch_progress(batch_idx, n_batches)

    model.train()   
    print()

    kpi_dict = model_kpi_func(
        p, kpi_input_dict,
        test_dataset.output_states_min, test_dataset.output_states_max,
        figure_name,
    )

    if eval_type == 'Test':
        with open(vis_data_path, 'wb') as fp:
            pickle.dump(kpi_input_dict, fp)

    return print_dict, kpi_dict