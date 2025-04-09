import torch
import torch.nn as nn

device = torch.device("cpu")

class CGRU_cell(nn.Module):
    """
    ConvGRU hücresi: GRU mantığını uzamsal (2D görüntü) veriye uygular.
    """

    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CGRU_cell, self).__init__()
        self.shape = shape  # Giriş boyutu (yükseklik, genişlik)
        self.input_channels = input_channels  # Giriş kanal sayısı (ör: 1 veya 3)
        self.filter_size = filter_size  # Konvolüsyon filtresi boyutu
        self.num_features = num_features  # Gizli katmandaki kanal sayısı
        self.padding = (
            filter_size - 1
        ) // 2  # Çıktının girişle aynı boyutta olması için padding

        # GRU kapılarını (z ve r) hesaplamak için konvolüsyon
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                self.input_channels + self.num_features,
                2 * self.num_features,
                self.filter_size,
                1,
                self.padding,
            ),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features),
        )

        # GRU'nun hücre çıktısı için konvolüsyon
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                self.input_channels + self.num_features,
                self.num_features,
                self.filter_size,
                1,
                self.padding,
            ),
            nn.GroupNorm(self.num_features // 32, self.num_features),
        )

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        # Başlangıç gizli durum yoksa sıfır tensorle başla
        if hidden_state is None:
            htprev = torch.zeros(
                inputs.size(1), self.num_features, self.shape[0], self.shape[1]
            ).to(device)
        else:
            htprev = hidden_state

        output_inner = []

        for index in range(seq_len):
            # Giriş yoksa (tahmin için), sıfır tensor kullan
            if inputs is None:
                x = torch.zeros(
                    htprev.size(0),
                    self.input_channels,
                    self.shape[0],
                    self.shape[1],
                ).to(device)
            else:
                x = inputs[index, ...]  # Zaman adımındaki giriş

            # Giriş ve önceki gizli durumu birleştir
            combined_1 = torch.cat((x, htprev), 1)

            # z ve r kapıları için konvolüsyon
            gates = self.conv1(combined_1)
            zgate, rgate = torch.split(gates, self.num_features, dim=1)

            # Kapılar sigmoid aktivasyonundan geçirilir
            z = torch.sigmoid(zgate)  # güncelleme kapısı
            r = torch.sigmoid(rgate)  # sıfırlama kapısı

            # r kapısı ile htprev'i zayıflat, giriş ile birleştir
            combined_2 = torch.cat((x, r * htprev), 1)
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)

            # Yeni gizli durum (GRU formülü)
            htnext = (1 - z) * htprev + z * ht

            output_inner.append(htnext)
            htprev = htnext  # bir sonraki adım için

        return torch.stack(output_inner), htnext


class CLSTM_cell(nn.Module):
    """
    ConvLSTM hücresi: LSTM mantığını 2D görüntülerde kullanır.
    """

    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2  # Çıkış boyutu korunur

        # Tek konvolüsyon üzerinden i, f, g, o (4 kapı) hesaplanır
        self.conv = nn.Sequential(
            nn.Conv2d(
                self.input_channels + self.num_features,
                4 * self.num_features,
                self.filter_size,
                1,
                self.padding,
            ),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features),
        )

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        if hidden_state is None:
            hx = torch.zeros(
                inputs.size(1), self.num_features, self.shape[0], self.shape[1]
            ).to(device)
            cx = torch.zeros(
                inputs.size(1), self.num_features, self.shape[0], self.shape[1]
            ).to(device)
        else:
            hx, cx = hidden_state

        output_inner = []

        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(
                    hx.size(0), self.input_channels, self.shape[0], self.shape[1]
                ).to(device)
            else:
                x = inputs[index, ...]

            # Giriş ve hx birleştirilir
            combined = torch.cat((x, hx), 1)

            # Tek seferde 4 kapıyı hesapla
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1
            )

            # LSTM kapıları
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            # Bellek durumu ve yeni gizli durum
            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)

            output_inner.append(hy)
            hx = hy
            cx = cy

        return torch.stack(output_inner), (hy, cy)
