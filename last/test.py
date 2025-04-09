import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Model
from models.conv_lstm import ConvLSTM_Model

from utils.gif_mp4 import *

# Dataset
from utils.dataloader import ImageDataset, ImageDatasetTest

# Video/frame fonksiyonları
from utils.video_extract import video_to_frames

# Yardımcı fonksiyonlar
from utils.utils import load_checkpoint


def test_model(test_loader, model, device):
    """
    MSE, MAE ve SSIM metriklerini hesaplar.
    """
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

                # SSIM hesaplaması için pencere boyutu
                min_dim = min(target.shape[-2], target.shape[-1])
                win_size = min(7, min_dim // 2 * 2 + 1)

                try:
                    # Eğer giriş RGB (3 kanal) ise, eksen belirleyelim
                    if target.shape[0] == 3:
                        ssim_value = ssim(
                            np.moveaxis(target, 0, -1),  # [C, H, W] -> [H, W, C]
                            np.moveaxis(output, 0, -1),
                            data_range=1.0,
                            win_size=win_size,
                            channel_axis=2
                        )
                    else:  # Grayscale (Tek Kanal)
                        ssim_value = ssim(
                            target.squeeze(),  # [1, H, W] -> [H, W]
                            output.squeeze(),
                            data_range=1.0,
                            win_size=win_size
                        )
                except ValueError as e:
                    print(f"SSIM hesaplama hatası: {e}. SSIM değeri 1.0 olarak atandı.")
                    ssim_value = 1.0  

                mse_total += mse
                mae_total += mae
                ssim_total += ssim_value

    num_samples = len(test_loader.dataset)

    if num_samples > 0:
        print(f"Test MSE: {mse_total / num_samples:.4f}")
        print(f"Test MAE: {mae_total / num_samples:.4f}")
        print(f"Test SSIM: {ssim_total / num_samples:.4f}")
    else:
        print("Test veri seti boş, metrik hesaplanamadı.")

    if num_samples > 0:
        print(f"Test MSE: {mse_total / num_samples:.4f}")
        print(f"Test MAE: {mae_total / num_samples:.4f}")
        print(f"Test SSIM: {ssim_total / num_samples:.4f}")
    else:
        print("Test veri seti boş, metrik hesaplanamadı.")


def test_wrong_movement(model, data_loader, device, threshold=0.005, visualize=False):
    """
    Anomali (yanlış hareket) tespiti yapar ve istenirse görselleri kaydeder.
    """
    model.eval()
    wrong_frames = []

    anomaly_save_dir = "./anomaly_images"
    os.makedirs(anomaly_save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (input_frames, target_frame) in enumerate(data_loader):
            input_frames = input_frames.to(device)
            target_frame = target_frame.to(device)

            outputs = model(input_frames)
            predicted_frame = outputs[:, -1, :, :, :]  

            mse_val = F.mse_loss(predicted_frame, target_frame, reduction='mean').item()
            is_anomaly = mse_val > threshold

            if is_anomaly:
                print(f"[Batch Index={idx}] Yanlış Hareket Tespit Edildi! (MSE={mse_val:.5f})")
                wrong_frames.append(idx)

                if visualize:
                    for b in range(predicted_frame.size(0)):
                        p_frame_np = predicted_frame[b].cpu().numpy().transpose(1, 2, 0)  
                        t_frame_np = target_frame[b].cpu().numpy().transpose(1, 2, 0)  

                        plt.figure(figsize=(8, 4))
                        plt.suptitle(f"Anomaly Detected! Batch {idx}, Sample {b}\nMSE={mse_val:.5f}", fontsize=12)

                        plt.subplot(1, 2, 1)
                        plt.imshow(t_frame_np)
                        plt.title("Gerçek Frame")
                        plt.axis("off")

                        plt.subplot(1, 2, 2)
                        plt.imshow(p_frame_np)
                        plt.title("Model Tahmini")
                        plt.axis("off")

                        plt.tight_layout()
                        save_name = f"anomaly_batch{idx}_sample{b}.png"
                        plt.savefig(os.path.join(anomaly_save_dir, save_name), dpi=300)
                        plt.close()

    print(f"\nToplam Anomalili Batch Sayısı: {len(wrong_frames)}")


def run_metrics(args):
    """
    'metrics' komutu için: MSE, MAE, SSIM metriklerini hesaplar.
    """
    frames_folder = "./test_data"
    video_to_frames(args.video, frames_folder)

    test_data = ImageDataset(image_folder=frames_folder, sequence_length=5, transform=None)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = ConvLSTM_Model(args).to(args.device)
    load_checkpoint(model, args, args.checkpoint)

    print("\n---- Test Metrics (MSE, MAE, SSIM) ----")
    test_model(test_loader, model, args.device)
    
    if args.save_gif:
        save_all_to_gif(test_loader, model, args.device, output_path="all_results.gif", frame_duration=0.1)
        print("GIF kaydedildi: all_results.gif")
        
    if args.save_mp4:
        gif_to_video_with_opencv("all_results.gif", "all_results.mp4", fps=10)
        print("MP4 kaydedildi: all_results.mp4")


def run_anomaly(args):
    """
    'anomaly' komutu için: Anomali tespiti yapar ve yanlış hareketleri kaydeder.
    """
    frames_folder = "./test_data"
    video_to_frames(args.image_folder, frames_folder)

    dataset_test = ImageDatasetTest(image_folder=frames_folder, sequence_length=args.sequence_length)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    model = ConvLSTM_Model(args).to(args.device)
    load_checkpoint(model, args, args.checkpoint)

    test_wrong_movement(model, test_loader, args.device, args.threshold, args.visualize)


def main():
    parser = argparse.ArgumentParser(description="ConvLSTM Model Test Script")

    subparsers = parser.add_subparsers(dest="command", help="Alt komutlar (metrics, anomaly)")

    parser_metrics = subparsers.add_parser("metrics", help="MSE, MAE, SSIM hesaplar.")
    parser_metrics.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser_metrics.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser_metrics.add_argument('--hidden_dim', type=int, default=64, help='ConvLSTM gizli boyutu')
    parser_metrics.add_argument('--input_dim', type=int, default=3, help='Giriş kanalı sayısı')
    parser_metrics.add_argument('--model', type=str, default='convlstm', help='Model ismi')
    parser_metrics.add_argument('--num_layers', type=int, default=4, help='ConvLSTM katman sayısı')
    parser_metrics.add_argument('--img_size', type=int, default=128, help='Görüntü boyutu (HxW)')
    parser_metrics.add_argument('--checkpoint', type=str, default="./model_ckpt/convlstm_layer4_best_model.pth",
                                help="Eğitilmiş model checkpoint yolu")
    parser_metrics.add_argument('--video', type=str, default="./correct.mp4", help="Test edilecek video dosyası")
    parser_metrics.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps'],
                                help='Cihaz seçimi (cpu veya mps)')
    parser_metrics.add_argument('--threshold', type=float, default=0.005, help='Anomali eşiği (kullanılmayabilir)')
    parser_metrics.add_argument('--save_gif', action='store_true', help="Test sürecini GIF olarak kaydet")
    parser_metrics.add_argument('--save_mp4', action='store_true', help="Test GIF'ini MP4 formatına dönüştür")


    parser_anomaly = subparsers.add_parser("anomaly", help="Anomali tespiti yapar.")
    parser_anomaly.add_argument('--image_folder', type=str, required=True,
                                help="Test için video veya klasör yolu (video_to_frames içinde kullanılır)")
    parser_anomaly.add_argument('--sequence_length', type=int, default=5,
                                help="ConvLSTM modeline girecek frame sayısı")
    parser_anomaly.add_argument('--checkpoint', type=str, required=True, 
                                help="Eğitilmiş model checkpoint yolu")
    parser_anomaly.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps'],
                                help="Modeli hangi cihazda çalıştıracağız")
    parser_anomaly.add_argument('--threshold', type=float, default=0.02, 
                                help="Yanlış hareket (anomali) MSE eşiği")
    parser_anomaly.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser_anomaly.add_argument('--hidden_dim', type=int, default=64, help="ConvLSTM gizli katman boyutu")
    parser_anomaly.add_argument('--input_dim', type=int, default=3, help="Giriş kanalı")
    parser_anomaly.add_argument('--num_layers', type=int, default=4, help="ConvLSTM katman sayısı")
    parser_anomaly.add_argument('--img_size', type=int, default=128, help="Görüntü boyutu")
    parser_anomaly.add_argument('--visualize', action='store_true',
                                help="Yanlış veya doğru frame’leri PNG olarak kaydetmek için kullanın")
    parser_anomaly.add_argument('--lr', type=float, default=1e-3, help="Modelin learning rate")

    args = parser.parse_args()

    if args.command == "metrics":
        run_metrics(args)
    elif args.command == "anomaly":
        run_anomaly(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()