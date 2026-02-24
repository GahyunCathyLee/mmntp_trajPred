import os
import sys
import time
import random

import numpy as np
import torch
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter

import Dataset
import params
import kpis
import top_functions
from evaluate import test_model_dict
import TPMs

def print_model_size(model):
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_count += param.nelement()
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"\n" + "="*50)
    print(f"📊 Model Size Info")
    print(f"  • Total Parameters : {param_count:,}")
    print(f"  • Model Memory Size: {size_all_mb:.2f} MB")
    print("="*50 + "\n")

def train_model_dict(p, prev_best_model=None, prev_itr=1):

    # ------------------------------------------------------------------
    # 디바이스 설정
    # ------------------------------------------------------------------
    if torch.cuda.is_available() and p.CUDA:
        device = torch.device("cuda:0")
        torch.cuda.manual_seed_all(0)
        print(f"✅  Device : {device}  ({torch.cuda.get_device_name(0)})")
    else:
        print("❌  CUDA not available — CPU mode is disabled. Exiting.")
        exit()

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')
    np.random.seed(1)
    random.seed(1)

    # ------------------------------------------------------------------
    # 모델 / 옵티마이저 초기화
    # ------------------------------------------------------------------
    model = p.model_dictionary['ref'](p.BATCH_SIZE, device, p.model_dictionary['hyperparams'], p)

    if p.TRANSFER_LEARNING == 'OnlyFC':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.trajectory_fc.parameters():
            param.requires_grad = True
        print("🔒  Transfer learning: only FC layer is trainable.")

    model = model.to(device)

    print_model_size(model)
    
    model = torch.compile(model)
    optimizer = p.model_dictionary['optimizer'](params=model.parameters(), lr=p.LR)

    if p.model_dictionary['hyperparams']['probabilistic output']:
        traj_loss_func = kpis.NLL_loss
    else:
        traj_loss_func = p.model_dictionary['traj loss function']()
    man_loss_func    = p.model_dictionary['man loss function']

    model_train_func = p.model_dictionary['model training function']
    model_eval_func  = p.model_dictionary['model evaluation function']
    model_kpi_func   = p.model_dictionary['model kpi function']

    print(f"📐  Model     : {type(model).__name__}")
    print(f"📐  Optimizer : {type(optimizer).__name__}  (lr={p.LR})")

    # ------------------------------------------------------------------
    # 데이터셋 로드
    # ------------------------------------------------------------------
    print("\n📂  Loading datasets...")

    tr_dataset = Dataset.LCDataset(
        p.TR.DATASET_DIR, p.TR.DATA_FILES,
        index_file=Dataset.get_index_file(p, p.TR, 'Tr'),
        data_type=p.model_dictionary['data type'],
        state_type=p.model_dictionary['state type'],
        use_map_features=p.hyperparams['model']['use_map_features'],
        keep_plot_info=False,
        force_recalc_start_indexes=False,
    )

    val_dataset = Dataset.LCDataset(
        p.TR.DATASET_DIR, p.TR.DATA_FILES,
        index_file=Dataset.get_index_file(p, p.TR, 'Val'),
        data_type=p.model_dictionary['data type'],
        state_type=p.model_dictionary['state type'],
        use_map_features=p.hyperparams['model']['use_map_features'],
        keep_plot_info=True,
        import_states=True,
        force_recalc_start_indexes=False,
        states_min=tr_dataset.states_min,
        states_max=tr_dataset.states_max,
        output_states_min=tr_dataset.output_states_min,
        output_states_max=tr_dataset.output_states_max,
    )

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
        output_states_max=tr_dataset.output_states_max,
    )

    print(f"   Train : {len(tr_dataset):>8,} samples")
    print(f"   Val   : {len(val_dataset):>8,} samples")
    print(f"   Test  : {len(te_dataset):>8,} samples")

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------
    prefix = "runs(debugging)" if p.DEBUG_MODE else "runs"
    tb_log_dir = f"{prefix}/{p.experiment_group}/{p.experiment_file}"
    tb = SummaryWriter(log_dir=tb_log_dir)
    print(f"\n📊  TensorBoard log : {tb_log_dir}")

    # ------------------------------------------------------------------
    # 학습 시작
    # ------------------------------------------------------------------
    print(f"\n🚀  Training starts  "
          f"(total_itrs={p.NUM_ITRS}, val_freq={p.VAL_FREQ}, "
          f"batch_size={p.BATCH_SIZE}, prev_itr={prev_itr})\n")

    val_result_dic = top_functions.train_top_func(
        p,
        model_train_func,
        model_eval_func,
        model_kpi_func,
        model,
        (traj_loss_func, man_loss_func),
        optimizer,
        tr_dataset,
        val_dataset,
        device,
        prev_best_model=prev_best_model,
        prev_itr=prev_itr,
        tensorboard=tb,
    )

    # ------------------------------------------------------------------
    # 결과 저장 (파라미터 튜닝 실험용 CSV)
    # ------------------------------------------------------------------
    if p.parameter_tuning_experiment:
        log_file_dir = os.path.join(p.TABLES_DIR, p.tuning_experiment_name + '.csv')
        log_columns  = ', '.join(p.log_dict.keys()) + '\n'
        result_line  = ', '.join(str(p.log_dict[k]) for k in p.log_dict) + '\n'
        if not os.path.exists(log_file_dir):
            result_line = log_columns + result_line
        with open(log_file_dir, 'a') as f:
            f.write(result_line)
        print(f"📝  Tuning result saved → {log_file_dir}")

    tb.close()


if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')

    p = params.ParametersHandler('MMnTP.yaml', 'highD.yaml', './config')
    p.hyperparams['experiment']['debug_mode'] = False
    p.match_parameters()

    p.IN_SEQ_LEN   = p.hyperparams['problem']['MAX_IN_SEQ_LEN']
    p.TGT_SEQ_LEN  = p.hyperparams['problem']['TGT_SEQ_LEN']
    p.MULTI_MODAL  = p.hyperparams['model']['multi_modal']
    p.MAN_DEC_IN   = p.hyperparams['model']['man_dec_in']
    p.MAN_DEC_OUT  = p.hyperparams['model']['man_dec_out']
    p.FEATURE_SIZE = 24

    train_model_dict(p)