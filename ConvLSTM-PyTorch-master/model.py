from torch import nn
import torch.nn.functional as F
import torch
from utils import make_layers

# -----------------------------
# Aktivasyon Fonksiyonu Sınıfı
# -----------------------------
class activation():
    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        # act_type: 'leaky', 'relu', 'sigmoid' gibi aktivasyon türleri
        self._act_type = act_type
        self.negative_slope = negative_slope  # LeakyReLU için alpha değeri
        self.inplace = inplace  # Bellek tasarrufu sağlar (ReLU için)

    def __call__(self, input):
        # sınıf çağrıldığında aktivasyon uygulanır
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError  # Bilinmeyen aktivasyon tipi girilirse hata verir


# -----------------------------
# Encoder-Decoder Yapısını Saran Ana Sınıf
# -----------------------------
class ED(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder  # Encoder modülü: Girdi frame'lerinden gizli durum üretir
        self.decoder = decoder  # Decoder modülü: Gizli durumlardan çıktılar (frame'ler) üretir

    def forward(self, input):
        state = self.encoder(input)     # Encoder çalıştırılır
        output = self.decoder(state)    # Elde edilen gizli durum decoder'a verilir
        return output                   # Sonuç (frame sekansı) döndürülür
