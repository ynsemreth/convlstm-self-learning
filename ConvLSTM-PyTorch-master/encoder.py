from torch import nn
from utils import make_layers
import torch
import logging

device = torch.device("cpu")

class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)  # Her subnet için bir RNN olmalı
        self.blocks = len(subnets)  # Kaç adet blok varsa

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index 1'den başlatılıyor (stage1, stage2 gibi isimler verilecek)
            setattr(self, 'stage' + str(index), make_layers(params))  # CNN katmanı
            setattr(self, 'rnn' + str(index), rnn)  # ConvLSTM veya ConvGRU

    def forward_by_stage(self, inputs, subnet, rnn):
        # inputs boyutu: (S, B, C, H, W)
        seq_number, batch_size, input_channel, height, width = inputs.size()

        # 5D -> 4D (CNN'e uygun format)
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)  # CNN katmanı (özellik çıkarımı)

        # 4D -> 5D
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1), inputs.size(2), inputs.size(3)))

        # RNN işlemi (ConvLSTM/GRU), başlangıç durumu None
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def forward(self, inputs):
        # inputs: (B, S, C, H, W) → (S, B, C, H, W)
        inputs = inputs.transpose(0, 1)
        hidden_states = []  # her katmandaki gizli durumlar burada toplanır

        logging.debug(inputs.size())

        for i in range(1, self.blocks + 1):
            # sırayla stage1 + rnn1, stage2 + rnn2 ... çalıştırılır
            inputs, state_stage = self.forward_by_stage(
                inputs,
                getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i))
            )
            hidden_states.append(state_stage)

        return tuple(hidden_states)  # decoder'a verilecek


if __name__ == "__main__":
    from net_params import convgru_encoder_params, convgru_decoder_params
    from data.mm import MovingMNIST

    # ConvGRU bazlı encoder oluşturuluyor
    encoder = Encoder(convgru_encoder_params[0], convgru_encoder_params[1]).to(device)

    # Veri seti hazırlanıyor
    trainFolder = MovingMNIST(is_train=True,
                              root='data/',
                              n_frames_input=10,
                              n_frames_output=10,
                              num_objects=[3])
    trainLoader = torch.utils.data.DataLoader(
        trainFolder,
        batch_size=4,
        shuffle=False,
    )
    # İlk batch ile encoder test ediliyor
    for i, (idx, targetVar, inputVar, _, _) in enumerate(trainLoader):
        inputs = inputVar.to(device)  # B, S, 1, 64, 64
        state = encoder(inputs)      # çıktı: gizli durumlar
        break