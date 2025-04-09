import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from utils.utils import *
from utils.video_extract import *
from utils.dataloader import *

def load_data():
    train_data = ImageDataset(image_folder="./dataset/train", sequence_length=5, transform=None)
    return train_data

def log_images(writer, x, y, logits, epoch):
    x_grid = make_grid(x[0], nrow=5)
    y_grid = make_grid(y[0], nrow=5)
    pred_grid = make_grid(logits[0], nrow=5)

    writer.add_image("Input Frames", x_grid, epoch)
    writer.add_image("Target Frames", y_grid, epoch)
    writer.add_image("Predicted Frames", pred_grid, epoch)

def calculate_accuracy(logits, targets):
    predictions = (logits > 0.5).float()
    correct_per_channel = (predictions == targets).float().mean(dim=(0, 2, 3, 4))
    return correct_per_channel.mean()

def main(args):
    start_epoch = 1
    best_loss = float('inf')
    lr = args.lr

    model = get_model(args)
    ckpt_path = f'./model_ckpt/{args.model}_layer{args.num_layers}_model.pth'
    ckpt_best_path = f'./model_ckpt/{args.model}_layer{args.num_layers}_best_model.pth'

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(ckpt_best_path), exist_ok=True)

    video_to_frames(args.video_dir, './dataset/train')

    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS desteklenmiyor. Apple Silicon cihazında çalıştırdığınızdan emin olun.")
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Desteklenmeyen cihaz: {args.device}")

    print(f"Using device: {device}")
    model.to(device)

    train_data = load_data()
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, drop_last=True)

    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if args.reload:
        try:
            start_epoch, lr, optimizer_state_dict = load_checkpoint(model, args, ckpt_path)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            optimizer.load_state_dict(optimizer_state_dict)
        except RuntimeError as e:
            print(f"Checkpoint yükleme hatası: {e}. Yeni model başlatılıyor.")

    writer = SummaryWriter(log_dir='./runs')

    for epoch in tqdm(range(start_epoch, args.epochs + 1), position=0):
        model.train()
        tq_train = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", total=len(train_loader), leave=False, position=1)

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for idx, (x, y) in enumerate(tq_train):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)

            if logits.shape[0] != y.shape[0]:  
                min_batch = min(logits.shape[0], y.shape[0])
                logits = logits[:min_batch]
                y = y[:min_batch]

            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += calculate_accuracy(logits, y).item()

            tq_train.set_postfix({'loss': '{:.03f}'.format(loss.item())})

        avg_loss = epoch_loss / len(train_loader)
        avg_accuracy = epoch_accuracy / len(train_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Train Accuracy = {avg_accuracy:.4f}")

        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Train/Accuracy", avg_accuracy, epoch)

        if epoch % 1 == 0 or epoch == 1:
            test_loss_avg = Averager()
            test_accuracy = 0.0
            model.eval()
            with torch.no_grad():
                tq_val = tqdm(train_loader, desc=f"Validation", total=len(train_loader), leave=False)
                for idx, (x, y) in enumerate(tq_val):
                    x, y = x.to(device), y.to(device)
                    logits = model(x)

                    if logits.shape[0] != y.shape[0]:  
                        min_batch = min(logits.shape[0], y.shape[0])
                        logits = logits[:min_batch]
                        y = y[:min_batch]

                    loss = loss_fn(logits, y)
                    test_loss_avg.add(loss.item())
                    test_accuracy += calculate_accuracy(logits, y).item()
                    tq_val.set_postfix(val_loss=f'{loss.item():.03f}')
                    
            log_images(writer, x, y, logits, epoch)
            avg_test_accuracy = test_accuracy / len(train_loader)
            val_loss = test_loss_avg.item()
            print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}, Validation Accuracy = {avg_test_accuracy:.4f}")

            writer.add_scalar("Validation/Loss", val_loss, epoch)
            writer.add_scalar("Validation/Accuracy", avg_test_accuracy, epoch)

            if best_loss > val_loss:
                best_loss = val_loss
                print(f"Epoch: {epoch}, Best loss: {best_loss:.4f}")
                save_checkpoint(model, optimizer, epoch, ckpt_best_path)

        save_checkpoint(model, optimizer, epoch, ckpt_path)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--hidden_dim', type=int, default=64, help='number of hidden dim for ConvLSTM layers')
    parser.add_argument('--input_dim', type=int, default=3, help='input channels')
    parser.add_argument('--model', type=str, default='convlstm', help='name of the model')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers')
    parser.add_argument('--gpu_num', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--img_size', type=int, default=128, help='image size')
    parser.add_argument('--reload', action='store_true', help='reload model')
    parser.add_argument('--video_dir', type=str, default='', help='root directory of the video')
    parser.add_argument('--seq_len', type=int, default=5, help='number of frames in a sequence')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps'], help='Device to use (cpu or mps)')
    args = parser.parse_args()

    main(args)