import torch
import torch.nn as nn
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(ConvLSTMCell, self).__init__()
        self.kernel_size = 3
        self.padding = 1
        self.hidden_dim = hidden_dim
        self.input_channels = input_channels

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels + hidden_dim,
                out_channels=4 * hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
            ),
            nn.GroupNorm(4 * hidden_dim, 4 * hidden_dim),
        )

    def forward(self, x, hidden):
        h, c = hidden

        if x.size(2) != h.size(2) or x.size(3) != h.size(3):
            diff_h = h.size(2) - x.size(2)
            diff_w = h.size(3) - x.size(3)
            x = torch.nn.functional.pad(x, (0, diff_w, 0, diff_h))

        conv_output = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = torch.chunk(conv_output, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)


class ConvLSTM_Model(nn.Module):
    def __init__(self, args):
        super(ConvLSTM_Model, self).__init__()
        self.batch_size = args.batch_size
        self.img_size = (args.img_size, args.img_size)
        self.n_layers = args.num_layers
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim

        self.cells = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(self.n_layers):
            input_dim = self.input_dim if i == 0 else self.hidden_dim
            self.cells.append(ConvLSTMCell(input_dim, self.hidden_dim))
            self.bns.append(nn.BatchNorm2d(self.hidden_dim))

        self.linear_conv = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.input_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, X, hidden=None, return_hidden=False):
        if hidden is None:
            hidden = self.init_hidden(X.size(0), self.img_size, X.device)

        for t in range(X.size(1)):
            inputs_x = X[:, t, :, :, :]
            for i, cell in enumerate(self.cells):
                inputs_x, hidden[i] = cell(inputs_x, hidden[i])
                inputs_x = self.bns[i](inputs_x)

        if return_hidden:
            return inputs_x, hidden

    # Prediction kısmı (örnek: video tahmini)
        predict = []
        inputs_x = X[:, -1, :, :, :] 
        for _ in range(X.size(1)): 
            for i, cell in enumerate(self.cells):
                inputs_x, hidden[i] = cell(inputs_x, hidden[i])
                inputs_x = self.bns[i](inputs_x)

            inputs_x = self.linear_conv(inputs_x)
            predict.append(inputs_x)

        predict = torch.stack(predict, dim=1)
        return torch.sigmoid(predict)

    def init_hidden(self, batch_size, img_size, device):
        h, w = img_size
        states = []
        for _ in range(self.n_layers):
            states.append((
                torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
                torch.zeros(batch_size, self.hidden_dim, h, w, device=device)
            ))
        return states

class ConvLSTM_RL_Agent(nn.Module):
    def __init__(self, convlstm_model, num_actions):
        super().__init__()
        self.convlstm_model = convlstm_model
        self.num_actions = num_actions

        # Hidden'dan gelen feature map'i flatten edip aksiyon çıkaracağız.
        feature_map_size = convlstm_model.hidden_dim * convlstm_model.img_size[0] * convlstm_model.img_size[1]
        self.fc = nn.Linear(feature_map_size, num_actions)

    def forward(self, X):
        batch_size = X.size(0)

        # ConvLSTM modelini çalıştır
        _, hidden_states = self.convlstm_model.forward(X, return_hidden=True)  # Bunu modelde ekleyeceğiz

        # Son katmanın son hidden state'i (h)
        last_hidden = hidden_states[-1][0]  # (h, c) tuple'ından h alınır

        # Flatten edip fully connected katmana ver
        flat = last_hidden.view(batch_size, -1)
        action_logits = self.fc(flat)

        return action_logits  # DQN için Q-value, PPO için logits olarak kullanılabilir
