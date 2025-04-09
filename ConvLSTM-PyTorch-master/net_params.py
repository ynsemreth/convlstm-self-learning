from collections import OrderedDict
from ConvRNN import CGRU_cell, CLSTM_cell

# ----------------------------------------
# ConvLSTM tabanlı model parametreleri
# ----------------------------------------
convlstm_encoder_params = [
    # CNN blokları (Encoder tarafı)
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),  # (C_in=1, C_out=16, kernel=3, stride=1, padding=1)
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],
    # ConvLSTM hücreleri (Encoder tarafı)
    [
        CLSTM_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=64),
        CLSTM_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=96),
        CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96),
    ]
]

convlstm_decoder_params = [
    # CNN blokları (Decoder tarafı)
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]  # son katman: tek kanal çıktıya dönüştür
        }),
    ],
    # ConvLSTM hücreleri (Decoder tarafı)
    [
        CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96),
        CLSTM_cell(shape=(32,32), input_channels=96, filter_size=5, num_features=96),
        CLSTM_cell(shape=(64,64), input_channels=96, filter_size=5, num_features=64),
    ]
]

# ----------------------------------------
# ConvGRU tabanlı model parametreleri
# ----------------------------------------
convgru_encoder_params = [
    # CNN blokları (Encoder tarafı)
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],
    # ConvGRU hücreleri (Encoder tarafı)
    [
        CGRU_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=64),
        CGRU_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=96),
        CGRU_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96)
    ]
]

convgru_decoder_params = [
    # CNN blokları (Decoder tarafı)
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],
    # ConvGRU hücreleri (Decoder tarafı)
    [
        CGRU_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96),
        CGRU_cell(shape=(32,32), input_channels=96, filter_size=5, num_features=96),
        CGRU_cell(shape=(64,64), input_channels=96, filter_size=5, num_features=64),
    ]
]