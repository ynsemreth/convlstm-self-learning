import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, image_folder, sequence_length=5, transform=None):
        self.image_folder = image_folder
        self.image_files = [
            os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.image_files.sort()
        self.sequence_length = sequence_length
        self.transform = transform

        if len(self.image_files) < self.sequence_length:
            raise ValueError(f"Klasörde yeterli görüntü dosyası yok: {len(self.image_files)} mevcut, ancak {self.sequence_length} gerekiyor.")

    def __len__(self):
        return len(self.image_files) - self.sequence_length 

    def __getitem__(self, idx):
        sequence_files = self.image_files[idx:idx + self.sequence_length + 1]

        frames = np.array([
            np.array(Image.open(file).convert("RGB"), dtype=np.float32) / 255.0 for file in sequence_files
        ])

        input_frames = frames[:-1]  
        target_frames = frames[1:]  

        input_frames = torch.tensor(input_frames).permute(0, 3, 1, 2)
        target_frames = torch.tensor(target_frames).permute(0, 3, 1, 2)

        return input_frames, target_frames

class ImageDatasetTest(Dataset):
    def __init__(self, image_folder, sequence_length=5, transform=None):
        self.image_folder = image_folder
        self.image_files = sorted([
            os.path.join(image_folder, f) 
            for f in os.listdir(image_folder) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.sequence_length = sequence_length
        self.transform = transform

        if len(self.image_files) < self.sequence_length + 1:
            raise ValueError(
                f"Klasörde yeterli frame yok! Toplam: {len(self.image_files)}, "
                f"gerekli: {self.sequence_length + 1}"
            )

    def __len__(self):
        return len(self.image_files) - self.sequence_length

    def __getitem__(self, idx):
        sequence_files = self.image_files[idx: idx + self.sequence_length + 1]
        
        frames = []
        for file_path in sequence_files:
            img = Image.open(file_path).convert("RGB") 
            img_np = np.array(img, dtype=np.float32) / 255.0  

            if self.transform:
                img_np = self.transform(img_np)

            frames.append(img_np)

        frames = np.array(frames)  

        input_frames = frames[:-1]  
        target_frame = frames[-1]  

        input_frames = torch.tensor(input_frames).permute(0, 3, 1, 2)
        target_frame = torch.tensor(target_frame).permute(2, 0, 1)  

        return input_frames, target_frame