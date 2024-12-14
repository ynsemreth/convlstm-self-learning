import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, image_folder, sequence_length=5, transform=None):
        """
        :param image_folder: Çerçevelerin bulunduğu klasör yolu
        :param sequence_length: Her bir örnek için ardışık çerçeve sayısı
        :param transform: Görseller için uygulanacak dönüşümler
        """
        self.image_folder = image_folder
        self.image_files = [
            os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.image_files.sort()  # Çerçeveleri sıralı yüklemek için
        self.sequence_length = sequence_length
        self.transform = transform

        if len(self.image_files) < self.sequence_length:
            raise ValueError(f"Klasörde yeterli görüntü dosyası yok: {len(self.image_files)} mevcut, ancak {self.sequence_length} gerekiyor.")

    def __len__(self):
        return len(self.image_files) - self.sequence_length  # Her bir ardışık çerçeve grubu bir örnek oluşturur

    def __getitem__(self, idx):
        sequence_files = self.image_files[idx:idx + self.sequence_length + 1]

        # Görselleri aç ve gri tonlamaya çevir, ardından NumPy dizisine çevir
        frames = np.array([
            np.array(Image.open(file).convert("L"), dtype=np.float32) / 255.0 for file in sequence_files
        ])

        # Giriş ve hedefi ayır
        input_frames = frames[:-1]  # İlk n çerçeve giriş
        target_frames = frames[1:]  # Son n çerçeve hedef

        # Tensöre çevir ve [T, C, H, W] formatına getir
        input_frames = torch.tensor(input_frames).unsqueeze(1)  # Zaman boyutu, kanal boyutu
        target_frames = torch.tensor(target_frames).unsqueeze(1)

        return input_frames, target_frames