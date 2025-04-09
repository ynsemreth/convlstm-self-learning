from torch import nn
from collections import OrderedDict

# -----------------------------------
# make_layers: Katmanları oluşturan fonksiyon
# -----------------------------------

def make_layers(block):
    layers = []  # oluşturulacak tüm katmanlar bu listede toplanır

    for layer_name, v in block.items():
        # layer_name: 'conv1_leaky_1' gibi isim
        # v: [in_channels, out_channels, kernel_size, stride, padding]

        if 'pool' in layer_name:
            # MaxPooling katmanı
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))

        elif 'deconv' in layer_name:
            # Transpose Convolution (up-sampling)
            transposeConv2d = nn.ConvTranspose2d(
                in_channels=v[0], out_channels=v[1],
                kernel_size=v[2], stride=v[3], padding=v[4]
            )
            layers.append((layer_name, transposeConv2d))

            # Aktivasyon fonksiyonu
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))

        elif 'conv' in layer_name:
            # Normal Convolution
            conv2d = nn.Conv2d(
                in_channels=v[0], out_channels=v[1],
                kernel_size=v[2], stride=v[3], padding=v[4]
            )
            layers.append((layer_name, conv2d))

            # Aktivasyon fonksiyonu
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))

        else:
            raise NotImplementedError  # Tanınmayan katman türü varsa hata ver

    # Sıralı katmanları nn.Sequential ile döndür
    return nn.Sequential(OrderedDict(layers))
