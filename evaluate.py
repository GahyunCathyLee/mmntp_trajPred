import os
import logging
from time import time
import numpy as np 
import pickle
import random

import torch
import torch.utils.data as utils_data
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt

import Dataset 
import TPMs 
import params
import top_functions
import kpis
import matplotlib.colors as mcolors

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def test_model_dict(p):
    # ------------------------------------------------------------------
    # 1. 디바이스 설정 (GPU 활성화 복구)
    # ------------------------------------------------------------------
    if torch.cuda.is_available() and p.CUDA:
        device = torch.device("cuda:0")
        torch.cuda.manual_seed_all(0)
        print(f"✅ Device : {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("⚠️ Device : CPU")
            
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    np.random.seed(1)
    random.seed(1)

    # ------------------------------------------------------------------
    # 2. 모델 및 초기화
    # ------------------------------------------------------------------
    model = p.model_dictionary['ref'](p.BATCH_SIZE, device, p.model_dictionary['hyperparams'], p)
    
    checkpoint_path = 'ckpts/baseline/best.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ Successfully loaded weights from {checkpoint_path}")
    
    model.eval()
    optimizer = p.model_dictionary['optimizer'](params=model.parameters(), lr=p.LR)
    man_loss_func = p.model_dictionary['man loss function']
    model_eval_func = p.model_dictionary['model evaluation function']
    model_kpi_func = p.model_dictionary['model kpi function']
    
    if p.model_dictionary['hyperparams']['probabilistic output']:
        traj_loss_func = kpis.NLL_loss
    else:
        traj_loss_func = p.model_dictionary['traj loss function']()

    # ------------------------------------------------------------------
    # 3. 데이터셋 로드
    # ------------------------------------------------------------------
    print("\n📂 Loading Train Dataset (for state normalization bounds)...")
    tr_dataset = Dataset.LCDataset(
        p.TR.DATASET_DIR, p.TR.DATA_FILES, 
        index_file=Dataset.get_index_file(p, p.TR, 'Tr'),
        data_type=p.model_dictionary['data type'], 
        state_type=p.model_dictionary['state type'], 
        use_map_features=p.hyperparams['model']['use_map_features'],
        keep_plot_info=False, 
        force_recalc_start_indexes=False
    )
    
    print("📂 Loading Test Dataset...")
    te_dataset = Dataset.LCDataset(
        p.TE.DATASET_DIR, p.TE.DATA_FILES,
        index_file=Dataset.get_index_file(p, p.TE, 'Te'),
        data_type=p.model_dictionary['data type'], 
        state_type=p.model_dictionary['state type'], 
        use_map_features=p.hyperparams['model']['use_map_features'],
        keep_plot_info=True, 
        import_states=True,
        force_recalc_start_indexes=False,
        states_min=tr_dataset.states_min, 
        states_max=tr_dataset.states_max,
        output_states_min=tr_dataset.output_states_min, 
        output_states_max=tr_dataset.output_states_max
    )
    
    # ------------------------------------------------------------------
    # 4. 모델 평가 및 KPI 추출
    # ------------------------------------------------------------------
    print("\n🚀 Starting Evaluation...")
    kpi_dict = top_functions.eval_top_func(
        p, model_eval_func, model_kpi_func, model, 
        (traj_loss_func, man_loss_func), te_dataset, device
    )
    
    p.export_evaluation(kpi_dict)

    # ------------------------------------------------------------------
    # 5. 원하는 Metric 터미널 출력 (ADE, FDE, RMSE @ t)
    # ------------------------------------------------------------------
    # 1. Losses
    print(f"\n[1. Losses]")
    loss_keys = ['Total Loss', 'Traj Loss', 'Man Loss', 'Mode Partial Loss', 'Man Partial Loss', 'Time Partial Loss']
    for key in loss_keys:
        val = kpi_dict.get(key, 0.0)
        print(f"  • {key:<20}: {val:.4f}")
        
    # 2. Manoeuvre Metrics (기동 예측 성능)
    print(f"\n[2. Manoeuvre Metrics]")
    acc = kpi_dict.get('minACC', {}).get('K=1', 0.0)
    mae = kpi_dict.get('minTimeMAE', {}).get('K=1', 0.0)
    rnd_mae = kpi_dict.get('minTimeMAE', {}).get('random', 0.0)
    print(f"  • Accuracy (K=1)    : {float(acc):.4f}")
    print(f"  • Time MAE (K=1)    : {float(mae):.4f} (Random: {float(rnd_mae):.4f})")
    
    modes = kpi_dict.get('activated modes group', [])
    probs = kpi_dict.get('activated modes percentage group', [])
    mode_str = " | ".join([f"Mode {m}: {p:.2f}%" for m, p in zip(modes, probs)])
    print(f"  • Mode Distribution : {mode_str}")
    
    # 3. Trajectory Metrics (m 단위)
    print(f"\n[3. Trajectory Metrics (meters)]")
    print(f"  • ADE               : {float(kpi_dict.get('ADE', 0)):.4f}")
    print(f"  • FDE               : {float(kpi_dict.get('FDE', 0)):.4f}")
    print(f"  • Total RMSE        : {float(kpi_dict.get('RMSE', 0)):.4f}")
    
    # 4. RMSE vs Horizon (Cumulative) - 논문 Table I 비교용
    print(f"\n[4. minRMSE-1 vs Prediction Horizon (Cumulative)]")
    min_rmse_df = kpi_dict.get('minRMSE', {}).get('K=1', None)
    if min_rmse_df is not None:
        try:
            # 데이터프레임에서 값만 추출하여 가로로 출력
            vals = min_rmse_df.values.flatten()
            headers = ["<1.0s", "<2.0s", "<3.0s", "<4.0s", "<5.0s"]
            header_row = " | ".join([f"{h:^10}" for h in headers])
            value_row  = " | ".join([f"{v:^10.4f}" for v in vals[:5]])
            print(f"    {header_row}")
            print(f"    {'-' * 60}")
            print(f"    {value_row}")
        except:
            print(f"    {min_rmse_df}")

    print("\n" + "="*65 + "\n")


if __name__ == '__main__':

    # ------------------------------------------------------------------
    # 6. 파라미터 핸들러 설정
    # ------------------------------------------------------------------
    p = params.ParametersHandler('MMnTP.yaml', 'highD.yaml', './config')
    
    p.hyperparams['experiment']['debug_mode'] = False
    
    p.hyperparams['dataset']['balanced'] = False
    p.hyperparams['training']['batch_size'] = 512
    p.hyperparams['experiment']['multi_modal_eval'] = True
    
    p.match_parameters()
    
    p.IN_SEQ_LEN   = p.hyperparams['problem']['MAX_IN_SEQ_LEN']
    p.TGT_SEQ_LEN  = p.hyperparams['problem']['TGT_SEQ_LEN']
    p.MULTI_MODAL  = p.hyperparams['model']['multi_modal']
    p.MAN_DEC_IN   = p.hyperparams['model']['man_dec_in']
    p.MAN_DEC_OUT  = p.hyperparams['model']['man_dec_out']
    p.FEATURE_SIZE = 24

    print(f"Evaluation Dataset DIR: {p.TE.DATASET_DIR}")
    test_model_dict(p)