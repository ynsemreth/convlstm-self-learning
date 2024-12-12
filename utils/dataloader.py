import os
import torch
from torch.utils.data import Dataset
import numpy as np

class NPYVideoDataset(Dataset):
    def __init__(self, root_dir, n_frames_input, n_frames_output, transform=None):
        self.root_dir = root_dir
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.transform = transform
        self.files = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_array = np.load(self.files[idx])

        if len(video_array) < (self.n_frames_input + self.n_frames_output):
            raise ValueError(f"Video array is too short: {len(video_array)} frames found, but {self.n_frames_input + self.n_frames_output} needed.")

        inputs = video_array[:self.n_frames_input]
        targets = video_array[self.n_frames_input:self.n_frames_input + self.n_frames_output]

        inputs = inputs[..., np.newaxis]
        targets = targets[..., np.newaxis]

        inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).float() / 255.0  
        targets = torch.from_numpy(targets).permute(0, 3, 1, 2).float() / 255.0

        if self.transform:
            inputs = self.transform(inputs)
            targets = self.transform(targets)

        return inputs, targets

