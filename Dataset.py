import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np 
import h5py
import matplotlib.pyplot as plt
import sys
import pickle
from random import shuffle
import math
import pdb
import time

class LCDataset(Dataset):
    def __init__(self, 
    dataset_dir, 
    data_files,
    data_type, 
    index_file,
        #'max_in_seq_len, out_seq_len, skip_seq_len,
        # _B for balanced(_U for unbalanced),tr_ratio,
        # abb_val_ratio,val_ratio,test_ratio,  
    state_type = '',
    use_map_features = False,
    keep_plot_info = True, 
    states_min = 0, 
    states_max = 0, 
    force_recalc_start_indexes = False,
    import_states = False, # import min-max of states
    output_states_min = 0, 
    output_states_max = 0,
    deploy_data = False):
        '''
        Index File Format: IndexGroup_InSeqLen_OutSeqLen_SkipSeqLen_BalancedIndex_TrRatio_AbbValRatio_ValRatio_TeRatio.npy
        IndexGroup Option: Tr, Val, Te, AbbTr, AbbVal, AbbTe
        BalancedIndex Option: B for balanced, U for unbalanced.
        '''
        super(LCDataset, self).__init__()
        
        self.data_files = data_files
        #print(data_files)
        #exit()
        self.index_file = index_file
        #print(index_file)
        self.main_dir = dataset_dir
        self.dataset_dirs = \
            [os.path.join(dataset_dir, data_file) for data_file in data_files]
        self.index_file_dirs = os.path.join(dataset_dir, self.index_file)
        self.file_size = []
        self.dataset_size = 0
        self.state_data_name = 'state_'+ state_type #+ '_data'
        self.data_type= data_type
        self.deploy_data = deploy_data
        self.parse_index_file()
        self.use_map_features = use_map_features
        
        
        self.keep_plot_info = keep_plot_info
        
        self.start_indexes = \
            self.get_samples_start_index(force_recalc_start_indexes)
        
        self.dataset_size = len(self.start_indexes)
        #print('{}: {}'.format(self.index_group, self.dataset_size))
       
        if data_type == 'state':
            if import_states == True:
                self.states_min = states_min
                self.states_max = states_max
            else:
                self.states_min, self.states_max = \
                    self.get_features_range(self.state_data_name)

            for i in range(len(self.states_min)):
                if self.states_min[i] == self.states_max[i]:
                    print('Warning! Feature {} min and max values are equal!'.format(i))
                    self.states_max[i] += 1
        else:
            self.states_min = 0
            self.states_max = 1  
        
        if import_states == True:
            self.output_states_min = output_states_min
            self.output_states_max = output_states_max
        else:
            self.output_states_min, self.output_states_max = \
                self.get_features_range('output_states_data')
        
        for i in range(len(self.output_states_min)):
            if self.output_states_min[i] == self.output_states_max[i]:
                self.output_states_max[i] += np.finfo('float').eps
    
        self.load_data()

    def __len__(self):
        return self.dataset_size

    def parse_index_file(self):
        index_data = self.index_file.split('_')
        try:
            self.index_group = index_data[0]
            self.min_in_seq_len = int(index_data[1])
            self.max_in_seq_len = int(index_data[2])
            self.out_seq_len = int(index_data[3]) if self.index_group != 'De' else 0
            #self.end_of_seq_skip_len = int(index_data[3])
            self.unbalanced_status = index_data[4]
            if index_data[4] == 'B':
                self.unbalanced = False
            elif index_data[4] == 'U':
                self.unbalanced = True
            else: 
                raise(ValueError('wrong dataset format'))
            self.tr_ratio = float(index_data[5])
            self.abb_val_ratio = float(index_data[6])
            self.val_ratio = float(index_data[7]) 
            self.te_ratio = float(index_data[8]) 
            self.de_ratio = float(index_data[9])
        except:
            print('Wrong index file format.')

    def get_features_range(self , feature_name):
        states_min = []
        states_max = []
        for dataset_dir in self.dataset_dirs:    
            with h5py.File(dataset_dir, 'r') as f:
                state_data = f[feature_name]
                states_min.append(np.min(state_data, axis = 0))
                states_max.append(np.max(state_data, axis = 0))
        
        states_min = np.stack(states_min, axis = 0)
        states_max = np.stack(states_max, axis = 0)
        
        states_min = states_min.min(axis = 0)
        states_max = states_max.max(axis = 0)
        
        #print('diff')
        #print(states_min)
        #print(states_max-states_min)
        #assert(np.any(states_max-states_min) != 0)
        
        #print('states_min:{}'.format(states_min))
        #print('states_max:{}'.format(states_max))
        #print('states_diff:{}'.format(states_max-states_min))
        #print('states_min_all:{}'.format(np.all(states_min)))
        #print('states_max_all:{}'.format(np.all(states_max)))
        
        return states_min, states_max
                
    def get_samples_start_index(self,force_recalc = False):
        if force_recalc or (not os.path.exists(self.index_file_dirs)):
            samples_start_index = []
                                
            print('Extracing start indexes for all index groups')
            valid_tv = -1
            
            # --- [수정] CS-LSTM과 동일한 샘플 추출을 위한 slide_step 설정 ---
            # 타겟 FPS가 5Hz이고 1초 간격으로 추출하므로 5로 고정하거나 파라미터로 받을 수 있습니다.
            slide_step = 5 
            # -----------------------------------------------------------------

            for file_itr, data_file in enumerate(self.dataset_dirs):
                print('File: {}'.format(data_file))
                with h5py.File(data_file, 'r') as f:
                    tv_ids = f['tv_data']
                    len_data_serie = tv_ids.shape[0]
                    current_tv = -1
                    
                    # --- [주의] 여기서 itr이 1프레임 단위로 돌던 것을 일단 유지하고,
                    # 아래 각 조건 분기문 안에서 시작점(itr)을 slide_step으로 제어해야 합니다. ---
                    
                    # (참고) MMnTP 원본 코드는 매우 복잡하게 분기되어 있습니다.
                    # 가장 일반적인 학습(Tr/Val 등) 시퀀스 추출 부분을 수정합니다.
                    
                    # 1) 전체 데이터를 순회하되, 차량(tv_id)이 바뀔 때마다 시작점을 초기화하고
                    # 2) 해당 차량 내에서만 slide_step 간격으로 샘플을 뽑도록 로직 변경

                    itr = 0
                    while itr < len_data_serie:
                        tv_id = tv_ids[itr]
                        
                        if self.index_group == 'Te':
                            # (Test 세트 로직은 원본 유지 또는 필요시 동일하게 수정)
                            if tv_id != current_tv:
                                sample_seq_len =  self.min_in_seq_len+self.out_seq_len
                                if (itr+sample_seq_len) <= len_data_serie:
                                    if np.all(tv_ids[itr:(itr+sample_seq_len)] == tv_id):
                                        samples_start_index.append([])
                                        valid_tv+=1
                                        current_tv = tv_id
                                        
                                        # --- [수정] Test 추출 시에도 slide_step 적용 ---
                                        for te_itr in range(itr+self.min_in_seq_len, len_data_serie-self.out_seq_len, slide_step):
                                            in_seq_len = min(te_itr-itr, self.max_in_seq_len)
                                            if np.all(tv_ids[te_itr:(te_itr+self.out_seq_len)] == tv_id):
                                                print('{}/{}/{}'.\
                                                      format(in_seq_len, te_itr-in_seq_len, len_data_serie), end = '\r')
                                                samples_start_index[valid_tv]\
                                                    .append([file_itr, te_itr-in_seq_len, in_seq_len])
                                            else:
                                                break
                                itr += 1 # 차량의 첫 시작점만 찾고 내부는 te_itr로 처리하므로 1칸 이동
                            else:
                                itr += 1

                        elif self.index_group == 'De':
                            # (Deploy 데이터 로직 - 생략/원본유지)
                            if tv_id != current_tv and (itr+self.min_in_seq_len) <= len_data_serie:
                                samples_start_index.append([])
                                valid_tv+=1
                                current_tv = tv_id
                                for te_itr in range(itr+self.min_in_seq_len, len_data_serie, slide_step):
                                    if tv_ids[te_itr-1] == tv_id and tv_ids[te_itr-self.min_in_seq_len] == tv_id:
                                        in_seq_len = min(te_itr-itr, self.max_in_seq_len)
                                        print('{}/{}/{}'.format(in_seq_len, te_itr-in_seq_len, len_data_serie), end = '\r')
                                        samples_start_index[valid_tv].append([file_itr, te_itr-in_seq_len, in_seq_len])
                                    else:
                                        break
                                itr += 1
                            else:
                                itr += 1
                                
                        else:
                            # --- [가장 중요한 일반 학습 데이터 추출 로직] ---
                            # CS-LSTM은 고정된 t_h(과거)를 사용하지만 MMnTP는 min_in_seq_len ~ max_in_seq_len을 가변적으로 허용함
                            # CS-LSTM과 맞추려면 max_in_seq_len에 해당하는 시퀀스만 뽑아야 함 (in_seq_len = max_in_seq_len)
                            
                            in_seq_len = self.max_in_seq_len 
                            sample_seq_len = in_seq_len + self.out_seq_len
                            
                            if (itr + sample_seq_len) <= len_data_serie:
                                if np.all(tv_ids[itr:(itr+sample_seq_len)] == tv_id):
                                    if tv_id != current_tv:
                                        samples_start_index.append([])
                                        valid_tv+=1
                                        current_tv = tv_id
                                    
                                    # 샘플 추가
                                    samples_start_index[valid_tv].append([file_itr, itr, in_seq_len])
                                    
                                    # --- [수정] 샘플을 찾았으면 slide_step 만큼 점프 ---
                                    itr += slide_step 
                                    print('{}/{}/{}'.format(in_seq_len, itr, len(tv_ids)), end = '\r')
                                else:
                                    # 연속성이 끊겼다면 1칸 이동해서 다시 확인
                                    itr += 1
                            else:
                                # 남은 길이가 부족하면 1칸 이동 (어차피 차량 바뀌면 다시 체크)
                                itr += 1                  
                        
                            
    
            samples_start_index = \
                [np.array(samples_start_index[itr]) for itr in range(len(samples_start_index))]
            shuffle(samples_start_index)
            

            n_tracks = len(samples_start_index)
            tr_samples = int(n_tracks*self.tr_ratio)
            abbVal_samples = int(n_tracks*self.abb_val_ratio)
            val_samples = int(n_tracks*self.val_ratio)
            te_samples = int(n_tracks*self.te_ratio)
            de_samples = int(n_tracks*self.de_ratio)
            index_groups = ['Tr', 'Val', 'AbbTe', 'Te', 'AbbTr', 'AbbVal', 'De']
            unbalanced_inds = ['U', 'B']
            
            start_indexes = {}
            start_indexes['B'] = {}
            start_indexes['U'] = {}
            
            start_indexes['U']['Tr'] = samples_start_index[:tr_samples]
            start_indexes['U']['Val'] = \
                samples_start_index[tr_samples:(tr_samples + val_samples)]
            start_indexes['U']['AbbTe'] = \
                samples_start_index[tr_samples:(tr_samples + val_samples)]
            start_indexes['U']['Te'] = \
                samples_start_index[(tr_samples + val_samples):\
                                    (tr_samples + val_samples + te_samples)]
            start_indexes['U']['AbbTr'] = \
                samples_start_index[abbVal_samples:tr_samples]
            start_indexes['U']['AbbVal'] = samples_start_index[:abbVal_samples]
            start_indexes['U']['De'] = \
                samples_start_index[(tr_samples+val_samples+ te_samples):\
                                    (tr_samples+val_samples+ te_samples+de_samples)]
            for index_group in index_groups: 
                print('Balancing {} dataset...'.format(index_group))
                #print(len(start_indexes['U'][index_group]))
                if len(start_indexes['U'][index_group]) == 0:
                    start_indexes['U'][index_group] = np.array([])
                    start_indexes['B'][index_group] = np.array([])
                else:
                    #print(index_group)
                    #print(start_indexes['U'][index_group])
                    start_indexes['U'][index_group] = \
                        np.concatenate(start_indexes['U'][index_group] , axis = 0)
                    start_indexes['B'][index_group] = \
                        self.balance_dataset(start_indexes['U'][index_group]) \
                            if index_group != 'De' else np.array([])
            
            for ub_ind in unbalanced_inds:
                index_file = modify_index_file(self.index_file, unbalanced_ind = ub_ind)
                for index_group in index_groups:
                    index_file = modify_index_file(index_file, index_group = index_group)
                    
                    index_file_dir = os.path.join(self.main_dir, index_file)
                    #samples_start_index = np.concatenate(start_indexes['B'][index_group] , axis = 0)
                    random_itrs = np.random.permutation(len(start_indexes[ub_ind][index_group]))
                    start_indexes[ub_ind][index_group] = start_indexes[ub_ind][index_group][random_itrs]
                    print('{}-{}: {}'.format(index_group, ub_ind, len(start_indexes[ub_ind][index_group])))
                    np.save(index_file_dir, start_indexes[ub_ind][index_group])
            
            samples_start_index = start_indexes[self.unbalanced_status][self.index_group]
                
        else:
            samples_start_index = np.load(self.index_file_dirs)
       
        return samples_start_index
    
    def balance_dataset(self, start_index):
        #force_recalc = True

        lc_count_in_lc_scenarios = 0
        lk_count_in_lc_scenarios = 0
        balanced_scenarios = np.zeros((len(start_index)))
        label_data_arr = []
        for dataset_dir in self.dataset_dirs:
            with h5py.File(dataset_dir, 'r') as f:
                labels_data = f['labels']
                labels_data = labels_data[:]
                label_data_arr.append(labels_data)
        
        for itr in range(len(start_index)):
            print('{}/{}'.format(itr, len(start_index)), end = '\r')
            file_itr = start_index[itr, 0]
            labels_data = label_data_arr[file_itr]
            start_itr = start_index[itr,1]
            in_seq_len = start_index[itr,2]
            label = abs(labels_data[(start_itr+in_seq_len):(start_itr + in_seq_len + self.out_seq_len)])    
            balanced_scenarios[itr] = np.any(label)*2 \
                # 2 is lc scenario, if there is a lc man at any time-step 
                # of sample, considered it in balanced dataset
            if np.any(label):
                lc_count_in_lc_scenarios += np.count_nonzero(label>0)
                lk_count_in_lc_scenarios += np.count_nonzero(label==0)
                  
        if lc_count_in_lc_scenarios> lk_count_in_lc_scenarios + self.out_seq_len:
            lk_balanced_count = int((lc_count_in_lc_scenarios-lk_count_in_lc_scenarios)/self.out_seq_len)
            lk_args = np.argwhere(balanced_scenarios == 0)
            lk_balanced_args = np.random.permutation(lk_args[:,0])[:lk_balanced_count]
            balanced_scenarios[lk_balanced_args] = 1 # 1 is balanced lk scenario
        

        return start_index[balanced_scenarios>0]

    def load_data(self):
        self.state_data = []
        self.frame_data = [] 
        self.tv_data = []
        self.output_data = [] 
        self.man_data = []
        
        start_time = time.time()
        for dataset_dir in self.dataset_dirs:
            with h5py.File(dataset_dir, 'r') as f:
                state_data_i = f[self.state_data_name][:]
                state_data_i = (state_data_i - self.states_min) / (self.states_max - self.states_min)
                self.state_data.append(torch.from_numpy(state_data_i.astype(np.float32)))
                
                output_data_i = f['output_states_data'][:]
                output_data_i = (output_data_i - self.output_states_min) / (self.output_states_max - self.output_states_min)
                self.output_data.append(torch.from_numpy(output_data_i.astype(np.float32)))
                
                self.man_data.append(torch.from_numpy(np.absolute(f['labels'][:].astype(np.int64))))
                self.frame_data.append(f['frame_data'][:])
                self.tv_data.append(f['tv_data'][:])
        
        end_time = time.time()
        #print('Data Loaded in {} sec'.format(end_time - start_time))

    def __getitem__(self, idx):
        file_itr = int(self.start_indexes[idx, 0])
        start_index = int(self.start_indexes[idx, 1])
        in_seq_len = int(self.start_indexes[idx, 2])
        pad_len = self.max_in_seq_len - in_seq_len

        # state
        states = self.state_data[file_itr][start_index:(start_index + in_seq_len)]
        if pad_len > 0:
            states = F.pad(states, (0, 0, pad_len, 0), 'constant', 0)
        
        padding_mask = torch.ones(self.max_in_seq_len, dtype=torch.bool)
        if pad_len > 0:
            padding_mask[:pad_len] = False  

        data_output = [states, padding_mask]

        # output states
        output_states = self.output_data[file_itr][start_index:(start_index + in_seq_len + self.out_seq_len)]
        if pad_len > 0:
            output_states = F.pad(output_states, (0, 0, pad_len, 0), 'constant', -1)
        data_output.append(output_states)

        # man
        man = self.man_data[file_itr][start_index:(start_index + in_seq_len + self.out_seq_len)]
        if pad_len > 0:
            man = F.pad(man, (pad_len, 0), 'constant', -1)

        # plot info
        if self.keep_plot_info:
            tv_id = self.tv_data[file_itr][start_index]
            frames = torch.from_numpy(
                self.frame_data[file_itr][start_index:(start_index + in_seq_len + self.out_seq_len)]
            )
            if pad_len > 0:
                frames = F.pad(frames, (pad_len, 0), 'constant', -1)
            plot_output = [tv_id, frames.numpy(), self.data_files[file_itr]]
        else:
            plot_output = ()

        return data_output, man, plot_output

def get_index_file(p, d_class, index_group):
    '''
        Index File Format: IndexGroup_InSeqLen_OutSeqLen_SkipSeqLen_BalancedIndex_TrRatio_AbbValRatio_ValRatio_TeRatio.npy
        IndexGroup Option: Tr, Val, Te, AbbTr, AbbVal, AbbTe
        BalancedIndex Option: B for balanced, U for unbalanced.
    '''
    if p.ABLATION:
        index_group = 'Abb' + index_group
    if p.UNBALANCED:
        unbalanced_ind = 'U'
    else:
        unbalanced_ind = 'B'
    index_file = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npy'\
        .format(index_group, p.MIN_IN_SEQ_LEN, p.MAX_IN_SEQ_LEN, p.TGT_SEQ_LEN,\
                 unbalanced_ind, d_class.TR_RATIO, d_class.ABBVAL_RATIO, \
                    d_class.VAL_RATIO, d_class.TE_RATIO, d_class.DE_RATIO, \
                        d_class.SELECTED_DATASET)
    return index_file

def modify_index_file(index_file,index_group = None, unbalanced_ind = None):
    
    index_list = index_file.split('_')
    if index_group is not None:
        index_list[0] = index_group
    if unbalanced_ind is not None:
        index_list[4] = unbalanced_ind
    index_file = '_'.join(index_list)
    return index_file
                