import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SignDataset(Dataset):
    def __init__(self, seq_dir, split='train', max_len=128, center_pad=True): # max_len limits frames in seq; center_pad pads sequences symmetrically; otherwise, pads at the end. 
        self.items = []
        self.seq_dir = seq_dir
        self.max_len = max_len
        self.center_pad = center_pad
        for fname in os.listdir(seq_dir):
            if not fname.endswith('.npz'):
                continue
            path = os.path.join(seq_dir, fname)
            data = np.load(path, allow_pickle=True)
            if data['split'].item() != split:
                continue
            self.items.append(path)

    def __len__(self):
        return len(self.items)

    def pad_or_trim(self, seq):
        T, V, C = seq.shape # number of frames, number of keypoints, coordinates per key point
        if T == self.max_len:
            return seq
        if T > self.max_len:
            start = (T - self.max_len)//2 if self.center_pad else 0
            return seq[start:start+self.max_len]
        
        # if shorter than max_len then pad
        pad_len = self.max_len - T
        pad = np.zeros((pad_len, V, C), dtype=seq.dtype)
        if self.center_pad:
            left = pad_len // 2
            right = pad_len - left
            seq = np.concatenate([pad[:left], seq, pad[:right]], axis=0)
        else:
            seq = np.concatenate([seq, pad], axis=0)
        return seq

    def __getitem__(self, idx):
        data = np.load(self.items[idx], allow_pickle=True)
        seq = data['seq'].astype(np.float32)  # [T, V, C]
        label = int(data['label'])
        seq = self.pad_or_trim(seq)
        # Normalize per-sequence (optional)
        # seq_mean = seq.mean(axis=(0,1), keepdims=True)
        # seq_std = seq.std(axis=(0,1), keepdims=True) + 1e-6
        # seq = (seq - seq_mean) / seq_std
        x = torch.from_numpy(seq).permute(2, 0, 1)  # [C, T, V]
        y = torch.tensor(label, dtype=torch.long)
        return x, y

# # test print
# dataset = SignDataset("data/sequences", split='train')
# x, y = dataset[0]
# print(x.shape, y)  # should give [C, T, V]; e.g., torch.Size([3, 128, 21]) 5 where 3 channels, 128 frames, 21 keypoints and 5 train samples.
