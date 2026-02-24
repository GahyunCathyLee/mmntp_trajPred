import os
import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm
import re
from concurrent.futures import ProcessPoolExecutor

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
FT_PER_M = 3.28084
LAT_LOOK_SEC = 4.0
LON_HIST_SEC = 3.0
LON_FUT_SEC  = 5.0

NEIGHBOR_COLS_8 = [
    "precedingId", "followingId", "leftPrecedingId", "leftAlongsideId",
    "leftFollowingId", "rightPrecedingId", "rightAlongsideId", "rightFollowingId"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="highD/raw")
    parser.add_argument("--out_dir", type=str, default="highD/processed_mmntp")
    parser.add_argument("--target_fps", type=float, default=5.0)
    parser.add_argument("--feature_mode", type=str, choices=['baseline', 'exp1', 'exp2', 'exp3', 'exp4'], default='baseline')
    
    parser.add_argument("--t_front", type=float, default=3.0)
    parser.add_argument("--t_back", type=float, default=5.0)
    parser.add_argument("--vy_eps", type=float, default=0.27)
    parser.add_argument("--eps_gate", type=float, default=0.1)
    
    return parser.parse_args()

def apply_upper_flip_like_npz(df: pd.DataFrame) -> pd.DataFrame:
    upper = df["drivingDirection"].fillna(2).astype(int) == 1
    if not upper.any(): return df
    
    x_max = float(df["x"].max())
    c_y = 2.0 * float(df.loc[upper, "y"].mean())
    upper_min_lane = int(df.loc[upper, "laneId"].min())
    upper_max_lane = int(df.loc[upper, "laneId"].max())

    df.loc[upper, "x"] = x_max - df.loc[upper, "x"].to_numpy()
    df.loc[upper, "y"] = c_y  - df.loc[upper, "y"].to_numpy()
    df.loc[upper, "laneId"] = (upper_min_lane + upper_max_lane) - df.loc[upper, "laneId"].astype(int)
    
    df.loc[upper, "xVelocity"] = -df.loc[upper, "xVelocity"]
    df.loc[upper, "yVelocity"] = -df.loc[upper, "yVelocity"]
    df.loc[upper, "xAcceleration"] = -df.loc[upper, "xAcceleration"]
    df.loc[upper, "yAcceleration"] = -df.loc[upper, "yAcceleration"]
    return df

def compute_lat_maneuver(lane_seq: np.ndarray, idx: int, w: int) -> int:
    """CUDA Error 방지를 위해 MMnTP 원래 방식(0-indexed) 적용"""
    lb = max(0, idx - w)
    ub = min(len(lane_seq) - 1, idx + w)
    if lane_seq[ub] > lane_seq[idx] or lane_seq[idx] > lane_seq[lb]:
        return 1 # Right
    if lane_seq[ub] < lane_seq[idx] or lane_seq[idx] < lane_seq[lb]:
        return 2 # Left
    return 0 # Keep

def extract_neighbor_features_flattened(ego_row, fast_lookup, args):
    ego_x, ego_y, ego_vx = ego_row['x'], ego_row['y'], ego_row['xVelocity']
    ego_frame = ego_row['frame']

    if args.feature_mode == 'baseline':
        selected_indices = [0, 1]  # dx, dy만 사용
    elif args.feature_mode == 'exp1':
        selected_indices = [6, 7]
    elif args.feature_mode == 'exp2':
        selected_indices = [4, 5, 6, 7, 8]
    elif args.feature_mode == 'exp3':
        selected_indices = [0, 1, 2, 3, 4, 5, 8]
    elif args.feature_mode == 'exp4':
        selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    nbr_features = np.zeros((8, len(selected_indices)), dtype=np.float32)
    vy_eps_ft = args.vy_eps * FT_PER_M 
    
    for i, col in enumerate(NEIGHBOR_COLS_8):
        nid = ego_row.get(col, 0)
        if pd.isna(nid) or nid == 0: continue
            
        nb_row = fast_lookup.get((ego_frame, nid))
        if nb_row is None: continue
        
        dx = nb_row['x'] - ego_x
        dy = nb_row['y'] - ego_y
        dvx = nb_row['xVelocity'] - ego_vx
        abs_vy = nb_row['yVelocity']
        
        vyn = abs_vy
        lc_state = 0.0
        if i >= 2:
            if i < 5:
                if vyn > vy_eps_ft: lc_state = -1.0
                elif vyn < -vy_eps_ft: lc_state = -3.0
                else: lc_state = -2.0
            else:
                if vyn < -vy_eps_ft: lc_state = 1.0
                elif vyn > vy_eps_ft: lc_state = 3.0
                else: lc_state = 2.0
        
        dx_time = dx / (dvx + (args.eps_gate if dvx >= 0 else -args.eps_gate))
        gate = 1.0 if (-args.t_back < dx_time < args.t_front) else 0.0
        
        full_feat = [dx, dy, dvx, abs_vy, nb_row['xAcceleration'], nb_row['yAcceleration'], lc_state, dx_time, gate]
        nbr_features[i] = [full_feat[idx] for idx in selected_indices]
        
    return nbr_features.flatten()

def process_recording(rec_id: str, raw_dir: Path, out_dir: Path, args):
    tracks_file = raw_dir / f"{rec_id}_tracks.csv"
    meta_file = raw_dir / f"{rec_id}_tracksMeta.csv"
    rec_meta_file = raw_dir / f"{rec_id}_recordingMeta.csv"
    
    if not (tracks_file.exists() and meta_file.exists() and rec_meta_file.exists()): return

    df = pd.read_csv(tracks_file)
    tmeta = pd.read_csv(meta_file)
    rmeta = pd.read_csv(rec_meta_file)
    
    raw_fps = float(rmeta.loc[0, "frameRate"])
    ds_stride = int(round(raw_fps / args.target_fps))
    
    df = df.merge(tmeta[["id", "drivingDirection"]], on="id", how="left")
    
    # Unit Conversion & Center Coordinates (CS-LSTM 동기화)
    df["xCenter_m"] = df["x"] + df["width"]  / 2.0
    df["yCenter_m"] = df["y"] + df["height"] / 2.0
    df["x"] = df["xCenter_m"] * FT_PER_M
    df["y"] = df["yCenter_m"] * FT_PER_M
    df["xVelocity"] = df["xVelocity"] * FT_PER_M
    df["yVelocity"] = df["yVelocity"] * FT_PER_M
    df["xAcceleration"] = df["xAcceleration"] * FT_PER_M
    df["yAcceleration"] = df["yAcceleration"] * FT_PER_M
    df["width"] = df["width"] * FT_PER_M
    df["height"] = df["height"] * FT_PER_M

    df["id"] = df["id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df["laneId"] = df["laneId"].astype(int)
    df["drivingDirection"] = df["drivingDirection"].astype(int)

    # Upper Flip 적용 (CS-LSTM 동기화)
    df = apply_upper_flip_like_npz(df)
    
    frame_min = int(df["frame"].min())
    df = df[((df["frame"] - frame_min) % ds_stride) == 0].copy()
    
    out_state_kine, out_output_states, out_labels, out_tv_data, out_frame_data = [], [], [], [], []
    
    t_h = int(round(LON_HIST_SEC * args.target_fps))
    t_f = int(round(LON_FUT_SEC * args.target_fps))
    w_frames = int(round(LAT_LOOK_SEC * args.target_fps))
    
    fast_lookup = df.set_index(['frame', 'id'])[['x', 'y', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']].to_dict('index')

    for vid, group in df.groupby("id"):
        group = group.sort_values("frame").reset_index(drop=True)
        
        # 최소 길이 보장 (과거 + 미래 데이터가 온전히 나올 수 있는 차량만)
        if len(group) < (t_h + 1 + t_f):
            continue
            
        lane_seq = group["laneId"].to_numpy()
        records = group.to_dict('records')
        
        # DataLoader가 시퀀스를 조립할 수 있도록 **모든 연속된 프레임**을 저장합니다.
        for idx, row in enumerate(records):
            lat_m = compute_lat_maneuver(lane_seq, idx, w_frames)
            target_xy = [row['x'], row['y']]
            
            base_feat = [
                row['x'], row['y'], row['xVelocity'], row['yVelocity'], 
                row['xAcceleration'], row['yAcceleration'], row['width'], row['height']
            ]
            
            nbr_feat = extract_neighbor_features_flattened(row, fast_lookup, args).tolist()
                
            out_state_kine.append(base_feat + nbr_feat)
            out_output_states.append(target_xy)
            out_labels.append(lat_m)
            out_tv_data.append(vid)
            out_frame_data.append(row['frame'])

    if not out_state_kine: return

    out_file = out_dir / f"{int(rec_id):02d}.h5"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('state_kine', data=np.array(out_state_kine, dtype=np.float32), compression="gzip")
        f.create_dataset('output_states_data', data=np.array(out_output_states, dtype=np.float32), compression="gzip")
        f.create_dataset('labels', data=np.array(out_labels, dtype=np.int64), compression="gzip")
        f.create_dataset('tv_data', data=np.array(out_tv_data, dtype=np.int32), compression="gzip")
        f.create_dataset('frame_data', data=np.array(out_frame_data, dtype=np.int32), compression="gzip")

def process_recording_wrapper(args_tuple): return process_recording(*args_tuple)

def main():
    args = parse_args()
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    rec_ids = sorted(set([re.match(r"(\d+)_tracks\.csv$", p.name).group(1) for p in raw_dir.glob("*_tracks.csv") if re.match(r"(\d+)_tracks\.csv$", p.name)]))
    
    print(f"Starting multiprocessing with {os.cpu_count()} cores...")
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_recording_wrapper, [(rec_id, raw_dir, out_dir, args) for rec_id in rec_ids]), total=len(rec_ids), desc="Processing HighD"))

if __name__ == "__main__":
    main()