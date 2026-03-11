import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import numpy as np
import re
from typing import Optional, Tuple, List


def extract_number(file_name: str) -> int:
    match = re.search(r"(\d+)", file_name)
    return int(match.group()) if match else 0


import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Tuple

class CustomVideoDataset(Dataset):
    def __init__(self, root_dir: str, flow_dir: str, text_emb, transform: Optional[callable] = None):
        
        self.root_dir = root_dir
        self.flow_dir = flow_dir
        self.transform = transform
        self.video_folders = sorted(os.listdir(root_dir), key=extract_number)
        self.text_emb = text_emb

    def __len__(self) -> int:
        return len(self.video_folders)

    def __getitem__(self, idx: int):
        video_name = self.video_folders[idx] # "video_41.pkl"
        pure_name = os.path.splitext(video_name)[0] 
        
        video_path = os.path.join(self.root_dir, video_name)
        with open(video_path, "rb") as f:
            video_data = pickle.load(f)
        
        features = video_data["feature"].astype("float32") # [C, L]
        e_labels = video_data["error_GT"]
        video_length = len(e_labels)
        text_pfe = self.text_emb

        flow_path = os.path.join(self.flow_dir, f"{pure_name}_flow6d.npy")
        
        if os.path.exists(flow_path):
            flow_mag = np.load(flow_path).astype("float32")
            flow_mag = torch.from_numpy(flow_mag)
        else:
            raise FileNotFoundError(f"flow not found: {flow_path}")

        if self.transform:
            features = self.transform(features)
        
        features = torch.from_numpy(features)

        return features, text_pfe, video_length, e_labels, flow_mag, pure_name

