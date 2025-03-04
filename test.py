import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Örnek model
from models.conv_lstm import ConvLSTM_Model

# Dataset
from utils.dataloader import ImageDataset, ImageDatasetTest

# Video/frame fonksiyonları
from utils.video_extract import video_to_frames
from utils.gif_mp4 import save_all_to_gif, gif_to_video_with_opencv

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
                ssim_value = ssim(
                    target.squeeze(),
                    output.squeeze(),
                    data_range=1.0,
                    win_size=5
                )

                mse_total += mse
                mae_total += mae
                ssim_total += ssim_value

    num_samples = len(test_loader.dataset)
    print(f"Test MSE: {mse_total / num_samples:.4f}")
    print(f"Test MAE: {mae_total / num_samples:.4f}")
    print(f"Test SSIM: {ssim_total / num_samples:.4f}")


def test_wrong_movement(model, data_loader, device, threshold=0.005, visualize=False):
    """
    Anomali (yanlış hareket) tespiti yapar. MSE eşiğinin üzerinde olan 
    frameleri "anomali" olarak işaretler. İstenirse sonuçlar görsel olarak kaydedilir.
    """
    model.eval()
    wrong_frames = []

    anomaly_save_dir = "./anomaly_images"
    normal_save_dir = "./normal_images"
    os.makedirs(anomaly_save_dir, exist_ok=True)
    os.makedirs(normal_save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (input_frames, target_frame) in enumerate(data_loader):
            input_frames = input_frames.to(device)
            target_frame = target_frame.to(device)

            outputs = model(input_frames)
            # outputs shape: [batch_size, seq_length, 1, H, W]
            predicted_frame = outputs[:, -1, :, :, :]  # Son frame tahmini

            mse_val = F.mse_loss(predicted_frame, target_frame, reduction='mean').item()
            is_anomaly = mse_val > threshold
            if is_anomaly:
                print(f"[Batch Index={idx}] Yanlış Hareket Tespit Edildi! (MSE={mse_val:.5f})")
                wrong_frames.append(idx)

            if visualize:
                batch_size_here = predicted_frame.size(0)
                for b in range(batch_size_here):
                    p_frame_np = predicted_frame[b].squeeze().cpu().numpy()
                    t_frame_np = target_frame[b].squeeze().cpu().numpy()

                    plt.figure(figsize=(8, 4))
                    if is_anomaly:
                        plt.suptitle(
                            f"Wrong Movement Detected! Batch Idx={idx}, Sample={b}\nMSE={mse_val:.5f}",
                            fontsize=12
                        )
                        file_prefix = "anomaly"
                        save_dir = anomaly_save_dir
                    else:
                        plt.suptitle(
                            f"Normal Movement: Batch Idx={idx}, Sample={b}\nMSE={mse_val:.5f}",
                            fontsize=12
                        )
                        file_prefix = "normal"
                        save_dir = normal_save_dir

                    plt.subplot(1, 2, 1)
                    plt.imshow(t_frame_np, cmap='gray')
                    plt.title("Gerçek Frame")
                    plt.axis("off")

                    plt.subplot(1, 2, 2)
                    plt.imshow(p_frame_np, cmap='gray')
                    plt.title("Model Tahmini")
                    plt.axis("off")

                    plt.tight_layout()
                    save_name = f"{file_prefix}_batch{idx}_sample{b}.png"
                    save_path = os.path.join(save_dir, save_name)
                    plt.savefig(save_path, dpi=300)
                    plt.close()

    print("\n----- Özet -----")
    total_frames = len(data_loader.dataset)
    total_wrong = len(wrong_frames)
    if total_frames == 0:
        print("Hiç frame işlenmedi.")
        return

    print(f"Toplam Batch Sayısı (Dataset Uzunluğu): {total_frames}")
    print(f"Anomalili Batch sayısı: {total_wrong}")
    print(f"Oran: %{(total_wrong / total_frames * 100):.2f}")


def run_metrics(args):
    """
    'metrics' alt komutu seçildiğinde çalışır.
    Metrix (MSE, MAE, SSIM) hesaplar, istenirse TensorBoard'a kaydeder, GIF & MP4 oluşturur.
    """
    # 1) Videoyu framelere ayır
    frames_folder = "./test_data"
    video_to_frames(args.video, frames_folder)

    # 2) DataLoader
    test_data = ImageDataset(
        image_folder=frames_folder, 
        sequence_length=5, 
        transform=None
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        shuffle=False
    )

    # 3) Model yükle
    model = ConvLSTM_Model(args).to(args.device)
    if args.checkpoint:
        load_checkpoint(model, args, args.checkpoint)

    # 4) TensorBoard
    writer = SummaryWriter("./runs")

    print("\n---- Test Metrics (MSE, MAE, SSIM) ----")
    test_model(test_loader, model, args.device)

    # 5) (Opsiyonel) Model mimarisini TensorBoard'a ekleyelim
    dummy_input = torch.randn(1, 5, args.input_dim, args.img_size, args.img_size).to(args.device)
    writer.add_graph(model, dummy_input)

    # 6) (Opsiyonel) GIF & MP4 kaydetme
    gif_path = "all_results.gif"
    video_path = "all_results.mp4"
    print("\n---- Saving GIF & MP4 ----")
    save_all_to_gif(
        test_loader, 
        model, 
        args.device, 
        output_path=gif_path, 
        frame_duration=0.1
    )
    gif_to_video_with_opencv(gif_path, video_path, fps=10)
    print(f"GIF saved at {gif_path}")
    print(f"Video saved at {video_path}")

    writer.close()


def run_anomaly(args):
    """
    'anomaly' alt komutu seçildiğinde çalışır.
    Anomali tespiti (yanlış hareket) yapar, istenirse görselleri kaydeder.
    """
    # 1) Videoyu framelere ayır
    frames_folder = "./test_data"
    video_to_frames(args.image_folder, frames_folder)

    # 2) DataLoader
    dataset_test = ImageDatasetTest(
        image_folder=frames_folder,
        sequence_length=args.sequence_length
    )
    test_loader = DataLoader(
        dataset_test, 
        batch_size=args.batch_size, 
        shuffle=False
    )

    # 3) Model
    model = ConvLSTM_Model(args).to(args.device)
    load_checkpoint(model, args, args.checkpoint)

    # 4) Yanlış hareket tespiti
    test_wrong_movement(
        model=model,
        data_loader=test_loader,
        device=args.device,
        threshold=args.threshold,
        visualize=args.visualize
    )


def main():
    parser = argparse.ArgumentParser(
        description="Tek dosyada hem metrik hesaplama hem anomali tespiti."
    )
    subparsers = parser.add_subparsers(dest="command", help="Alt komutlar (metrics veya anomaly)")

    # -------------------------------------------------------------
    # 1) METRICS subcommand
    # -------------------------------------------------------------
    parser_metrics = subparsers.add_parser("metrics", help="MSE, MAE, SSIM hesaplar, GIF/MP4 kaydeder, vs.")

    parser_metrics.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser_metrics.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser_metrics.add_argument('--hidden_dim', type=int, default=64, help='ConvLSTM gizli boyutu')
    parser_metrics.add_argument('--input_dim', type=int, default=1, help='Giriş kanalı sayısı')
    parser_metrics.add_argument('--model', type=str, default='convlstm', help='Model ismi')
    parser_metrics.add_argument('--num_layers', type=int, default=4, help='ConvLSTM katman sayısı')
    parser_metrics.add_argument('--img_size', type=int, default=64, help='Görüntü boyutu (HxW)')
    parser_metrics.add_argument('--checkpoint', type=str, default="./model_ckpt/convlstm_layer4_model.pth",
                                help="Eğitilmiş model checkpoint yolu")
    parser_metrics.add_argument('--video', type=str, default="./train.mp4", help="Test edilecek video dosyası")
    parser_metrics.add_argument('--device', type=str, default='mps', choices=['cpu', 'mps'],
                                help='Cihaz seçimi (cpu veya mps)')
    parser_metrics.add_argument('--threshold', type=float, default=0.005, help='Anomali eşiği (kullanılmayabilir)')

    # -------------------------------------------------------------
    # 2) ANOMALY subcommand
    # -------------------------------------------------------------
    parser_anomaly = subparsers.add_parser("anomaly", help="Anomali (yanlış hareket) tespiti yapar.")

    parser_anomaly.add_argument('--image_folder', type=str, required=True,
                                help="Test için video veya klasör yolu (video_to_frames içinde kullanılır)")
    parser_anomaly.add_argument('--sequence_length', type=int, default=5,
                                help="ConvLSTM modeline girecek frame sayısı")
    parser_anomaly.add_argument('--checkpoint', type=str, required=True, 
                                help="Eğitilmiş model checkpoint yolu")
    parser_anomaly.add_argument('--device', type=str, default='mps', choices=['cpu', 'mps'],
                                help="Modeli hangi cihazda çalıştıracağız")
    parser_anomaly.add_argument('--threshold', type=float, default=0.02, 
                                help="Yanlış hareket (anomali) MSE eşiği")
    parser_anomaly.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser_anomaly.add_argument('--hidden_dim', type=int, default=64, help="ConvLSTM gizli katman boyutu")
    parser_anomaly.add_argument('--input_dim', type=int, default=1, help="Giriş kanalı")
    parser_anomaly.add_argument('--num_layers', type=int, default=2, help="ConvLSTM katman sayısı")
    parser_anomaly.add_argument('--img_size', type=int, default=64, help="Görüntü boyutu")
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
