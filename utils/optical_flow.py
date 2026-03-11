import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

def extract_6d_flow_from_frames(video_dir):
    frame_paths = sorted(
        glob(os.path.join(video_dir, "*.jpg")) +
        glob(os.path.join(video_dir, "*.png"))
    )
    T = len(frame_paths)
    if T < 2:
        return None

    feats = []

    prev = cv2.imread(frame_paths[0])
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    prev_mag = None
    prev_ang = None

    for i in range(1, T):
        curr = cv2.imread(frame_paths[i])
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        u = flow[..., 0]
        v = flow[..., 1]

        mag = np.sqrt(u ** 2 + v ** 2)
        ang = np.arctan2(v, u)

        mean_u = np.mean(u)
        mean_v = np.mean(v)
        mean_mag = np.mean(mag)
        mean_ang = np.mean(ang)

        if prev_mag is None:
            d_mag = 0.0
            d_ang = 0.0
        else:
            d_mag = mean_mag - prev_mag
            d_ang = mean_ang - prev_ang

        feats.append([
            mean_u,
            mean_v,
            mean_mag,
            d_mag,
            mean_ang,
            d_ang
        ])

        prev_gray = curr_gray
        prev_mag = mean_mag
        prev_ang = mean_ang

    feats = np.array(feats, dtype=np.float32)      # [T-1, 6]
    feats = np.concatenate([feats, feats[-1:]], 0) # 补齐到 T
    return feats.T                                  # [6, T]
def process_dataset(root_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    video_dirs = sorted([
        d for d in glob(os.path.join(root_dir, "video*"))
        if os.path.isdir(d)
    ])

    print(f"Found {len(video_dirs)} videos in {root_dir}")

    for vdir in tqdm(video_dirs):
        name = os.path.basename(vdir)
        save_path = os.path.join(save_dir, f"{name}_flow6d.npy")

        if os.path.exists(save_path):
            continue

        feat = extract_6d_flow_from_frames(vdir)
        if feat is None:
            continue

        np.save(save_path, feat)
if __name__ == "__main__":
    TRAIN_ROOT = "./data/sar-train-frames"
    TEST_ROOT  = "./data/sar-test-frames"

    SAVE_TRAIN = "./data/flow6d_train"
    SAVE_TEST  = "./data/flow6d_test"

    # process_dataset(TRAIN_ROOT, SAVE_TRAIN)
    process_dataset(TEST_ROOT, SAVE_TEST)

