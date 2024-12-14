import argparse
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from models.conv_lstm import ConvLSTM_Model
from utils.dataloader import ImageDataset
from utils.video_extract import video_to_frames
from utils.gif_mp4 import *


def test_model(test_loader, model, device):
    model.eval()
    mse_total, mae_total, ssim_total = 0.0, 0.0, 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for target, output in zip(targets_np, outputs_np):
                mse = np.mean((target - output) ** 2)
                mae = np.mean(np.abs(target - output))
                ssim_value = ssim(target.squeeze(), output.squeeze(), data_range=1.0, win_size=5)

                mse_total += mse
                mae_total += mae
                ssim_total += ssim_value

    num_samples = len(test_loader.dataset)
    print(f"Test MSE: {mse_total / num_samples:.4f}")
    print(f"Test MAE: {mae_total / num_samples:.4f}")
    print(f"Test SSIM: {ssim_total / num_samples:.4f}")

if __name__ == "__main__":
    gif_path = "all_results.gif"
    video_path = "all_results.mp4"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--model', type=str, default='convlstm')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--checkpoint', type=str, default="./model_ckpt/convlstm_layer4_model.pth")
    parser.add_argument('--video', type=str, default="./train1.mp4")
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'mps'], help='Device to use (cpu or mps)')
    args = parser.parse_args([])

    frames_folder = "./test_data"
    video_to_frames(args.video, frames_folder)

    test_data = ImageDataset(image_folder=frames_folder, sequence_length=5, transform=None)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = ConvLSTM_Model(args).to(args.device)
    from utils.utils import load_checkpoint
    if args.checkpoint:
        load_checkpoint(model, args, args.checkpoint)

    test_model(test_loader, model, args.device)
    save_all_to_gif(test_loader, model, args.device, output_path=gif_path, frame_duration=0.1)
    gif_to_video_with_opencv(gif_path, video_path, fps=10)
