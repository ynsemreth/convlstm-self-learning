import torch
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from models.conv_lstm import ConvLSTM_Model
import imageio

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

def save_all_to_gif(test_loader, model, device, output_path="all_results.gif"):
    model.eval()
    input_frames, target_frames, output_frames = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            input_frames.extend(((inputs.cpu().numpy() * 255).clip(0, 255)).astype(np.uint8).squeeze(2))
            target_frames.extend(((targets.cpu().numpy() * 255).clip(0, 255)).astype(np.uint8).squeeze(2))
            output_frames.extend(((outputs.cpu().numpy() * 255).clip(0, 255)).astype(np.uint8).squeeze(2))

    with imageio.get_writer(output_path, mode="I", duration=0.5) as writer:
        for i in range(len(input_frames)):
            combined_frame = np.concatenate(
                [input_frames[i], target_frames[i], output_frames[i]], axis=1
            )
            writer.append_data(combined_frame)

    print(f"Saved combined GIF: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--model', type=str, default='convlstm')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--checkpoint', type=str, default="./model_ckpt/convlstm_layer4_model.pth")
    parser.add_argument('--test_data', type=str, default="./dataset/train")
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'mps'], help='Device to use (cpu or mps)')
    args = parser.parse_args([])

    from utils.dataloader import *
    test_data = ImageDataset(image_folder=args.test_data, sequence_length=5 , transform=None)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = ConvLSTM_Model(args).to(args.device)
    from utils.utils import load_checkpoint
    if args.checkpoint:
        load_checkpoint(model, args, args.checkpoint)

    test_model(test_loader, model, args.device)

    save_all_to_gif(test_loader, model, args.device, output_path="all_results.gif")
