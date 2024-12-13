import os
import torch
from torch.utils.data import Dataset
import numpy as np

class NPYVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_array = np.load(self.files[idx])

        # Toplam kare sayısını kontrol et
        total_frames = len(video_array)
        if total_frames < 2:
            raise ValueError(f"Video array is too short: {total_frames} frames found, at least 2 needed.")

        # Tüm kareleri dinamik olarak ayarla
        midpoint = total_frames // 2
        inputs = video_array[:midpoint]
        targets = video_array[midpoint:]

        # Kanal boyutunu ekle ve tensör formatına çevir
        inputs = inputs[..., np.newaxis]
        targets = targets[..., np.newaxis]

        inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).float() / 255.0
        targets = torch.from_numpy(targets).permute(0, 3, 1, 2).float() / 255.0

        if self.transform:
            inputs = self.transform(inputs)
            targets = self.transform(targets)

        return inputs, targets
