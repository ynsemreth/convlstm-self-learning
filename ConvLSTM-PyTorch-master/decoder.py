from torch import nn
from utils import make_layers
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)  # subnet ve rnn sayısı eşit olmalı

        self.blocks = len(subnets)  # toplam katman sayısı

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            # katmanları rnn3, rnn2, ... ve stage3, stage2, ... olarak ayarla
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index), make_layers(params))

    def forward_by_stage(self, inputs, state, subnet, rnn):
        # her zaman adımında ConvRNN çalıştırılır
        inputs, state_stage = rnn(inputs, state, seq_len=10)

        # 5D çıktıyı (S, B, C, H, W) 4D’ye (S*B, C, H, W) çevir
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))

        # CNN ile işle, sonra tekrar 5D’ye çevir
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs

    def forward(self, hidden_states):
        # En son katmandan başlanır (en yüksek seviye gizli durum)
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage3'),
                                       getattr(self, 'rnn3'))

        # Geriye doğru giderek her katmanda işler (3 -> 2 -> 1)
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))

        # boyut: (S, B, C, H, W) --> (B, S, C, H, W)
        inputs = inputs.transpose(0, 1)
        return inputs


if __name__ == "__main__":
    from net_params import convlstm_encoder_params, convlstm_forecaster_params
    from data.mm import MovingMNIST
    from encoder import Encoder

    encoder = Encoder(convlstm_encoder_params[0],
                      convlstm_encoder_params[1]).to(device)
    decoder = Decoder(convlstm_forecaster_params[0],
                      convlstm_forecaster_params[1]).to(device)

    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)

    trainFolder = MovingMNIST(is_train=True,
                              root='data/',
                              n_frames_input=10,
                              n_frames_output=10,
                              num_objects=[3])
    trainLoader = torch.utils.data.DataLoader(
        trainFolder,
        batch_size=8,
        shuffle=False,
    )
    for i, (idx, targetVar, inputVar, _, _) in enumerate(trainLoader):
        inputs = inputVar.to(device)  # B,S,1,64,64
        state = encoder(inputs)
        break

    output = decoder(state)
    print(output.shape)  # S,B,1,64,64
