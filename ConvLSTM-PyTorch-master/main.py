import os
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from data.mm import MovingMNIST
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse

# ----------------------
# Argparse: Parametreler
# ----------------------
TIMESTAMP = "2025-04-10T00-00-00"
parser = argparse.ArgumentParser()
parser.add_argument('-clstm', '--convlstm', help='use convlstm as base cell', action='store_true')
parser.add_argument('-cgru', '--convgru', help='use convgru as base cell', action='store_true')
parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('-frames_input', default=10, type=int, help='number of input frames')
parser.add_argument('-frames_output', default=10, type=int, help='number of frames to predict')
parser.add_argument('-epochs', default=1, type=int, help='number of training epochs')
args = parser.parse_args()

# ----------------------
# Reproducibility (Sabitlik)
# ----------------------
random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# ----------------------
# Veri Hazırlığı
# ----------------------
save_dir = './save_model/' + TIMESTAMP
trainFolder = MovingMNIST(is_train=True, root='data/', n_frames_input=args.frames_input, n_frames_output=args.frames_output, num_objects=[3])
validFolder = MovingMNIST(is_train=False, root='data/', n_frames_input=args.frames_input, n_frames_output=args.frames_output, num_objects=[3])
trainLoader = torch.utils.data.DataLoader(trainFolder, batch_size=args.batch_size, shuffle=False)
validLoader = torch.utils.data.DataLoader(validFolder, batch_size=args.batch_size, shuffle=False)

# ----------------------
# Model Parametre Seçimi
# ----------------------
if args.convlstm:
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
elif args.convgru:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
else:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params

device = torch.device("cpu")

# ----------------------
# Eğitim Fonksiyonu
# ----------------------
def train():
    encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
    decoder = Decoder(decoder_params[0], decoder_params[1]).to(device)
    net = ED(encoder, decoder)

    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)  # TensorBoard loglama

    early_stopping = EarlyStopping(patience=20, verbose=True)
    net = nn.DataParallel(net)
    net.to(device)

    cur_epoch = 0
    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    lossfunction = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    for epoch in range(cur_epoch, args.epochs + 1):
        # ------------------------
        # Eğitim Döngüsü
        # ------------------------
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            inputs = inputVar.to(device)
            label = targetVar.to(device)
            optimizer.zero_grad()
            net.train()
            pred = net(inputs)
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix({'trainloss': '{:.6f}'.format(loss_aver), 'epoch': '{:02d}'.format(epoch)})
        tb.add_scalar('TrainLoss', loss_aver, epoch)

        # ------------------------
        # Doğrulama Döngüsü
        # ------------------------
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
                if i == 3000:
                    break
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                pred = net(inputs)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / args.batch_size
                valid_losses.append(loss_aver)
                t.set_postfix({'validloss': '{:.6f}'.format(loss_aver), 'epoch': '{:02d}'.format(epoch)})
        tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cpu.empty_cache()

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print(f'[{epoch}/{args.epochs}] train_loss: {train_loss:.6f} valid_loss: {valid_loss:.6f}')

        train_losses = []
        valid_losses = []

        pla_lr_scheduler.step(valid_loss)

        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        early_stopping(valid_loss, model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Kayıt
    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)

if __name__ == "__main__":
    train()