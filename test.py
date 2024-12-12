import torch
import argparse
import numpy as np
from utils.utils import load_checkpoint
from utils.dataloader import NPYVideoDataset
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from models.conv_lstm import ConvLSTM_Model
import imageio

def test_model(test_loader, model, device):
    """Test model and calculate metrics (MSE, MAE, SSIM)."""
    model.eval()
    mse_total, mae_total, ssim_total = 0.0, 0.0, 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Convert to NumPy for metric calculations
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for target, output in zip(targets_np, outputs_np):
                mse = np.mean((target - output) ** 2)
                mae = np.mean(np.abs(target - output))
                ssim_value = ssim(target.squeeze(), output.squeeze(), data_range=1.0)

                mse_total += mse
                mae_total += mae
                ssim_total += ssim_value

    num_samples = len(test_loader.dataset)
    print(f"Test MSE: {mse_total / num_samples:.4f}")
    print(f"Test MAE: {mae_total / num_samples:.4f}")
    print(f"Test SSIM: {ssim_total / num_samples:.4f}")


def save_all_to_gif(test_loader, model, device, output_path="all_results.gif"):
    """Generate and save a single GIF for the entire dataset."""
    model.eval()
    input_frames, target_frames, output_frames = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            input_frames.extend((inputs.cpu().numpy() * 255).astype(np.uint8).squeeze(2))
            target_frames.extend((targets.cpu().numpy() * 255).astype(np.uint8).squeeze(2))
            output_frames.extend((outputs.cpu().detach().numpy() * 255).astype(np.uint8).squeeze(2))

    # Combine all frames into a single GIF
    with imageio.get_writer(output_path, mode="I", duration=0.5) as writer:
        for i in range(len(input_frames)):
            combined_frame = np.concatenate(
                [input_frames[i], target_frames[i], output_frames[i]], axis=1
            )
            writer.append_data(combined_frame)

    print(f"Saved combined GIF: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension for ConvLSTM layers')
    parser.add_argument('--input_dim', type=int, default=1, help='input channels')
    parser.add_argument('--model', type=str, default='convlstm', help='model name')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--frame_num', type=int, default=10, help='number of frames')
    parser.add_argument('--img_size', type=int, default=64, help='image size')
    parser.add_argument('--checkpoint', type=str, default="./model_ckpt/convlstm_layer2_best_model.pth", help='path to checkpoint')
    parser.add_argument('--test_data', type=str, default="./test", help='path to test dataset')
    args = parser.parse_args([])

    # Dataset and DataLoader
    test_data = NPYVideoDataset(
        root_dir=args.test_data,
        n_frames_input=args.frame_num,
        n_frames_output=args.frame_num
    )
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # Load model and checkpoint
    model = ConvLSTM_Model(args).to(device)
    if args.checkpoint:
        start_epoch, args.lr, optimizer_state_dict = load_checkpoint(model, args, args.checkpoint)

    # Test the model
    test_model(test_loader, model, device)

    # Generate and save combined GIF
    save_all_to_gif(test_loader, model, device, output_path="all_results.gif")
