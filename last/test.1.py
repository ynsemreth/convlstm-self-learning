import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.utils import get_model  # Model oluşturma fonksiyonu
from utils.dataloader import ImageDatasetTest  # Test veri setini sağlayan sınıf

def main(args):
    # Sonuçların kaydedileceği klasörü oluştur
    os.makedirs("./results", exist_ok=True)
    
    # Cihaz ayarlaması (CPU veya MPS)
    device = torch.device(args.device)
    print(f"Kullanılan cihaz: {device}")

    # Modeli oluştur ve cihaza taşı
    model = get_model(args)
    model.to(device)
    model.eval()  

    # Checkpoint dosyasını yükle: Eğitilmiş modelin ağırlıklarını kullanıyoruz.
    ckpt_path = f'./model_ckpt/{args.model}_layer{args.num_layers}_model.pth'
    if os.path.exists(ckpt_path):
        try:
            # Checkpoint dosyasını yükleyip modelin ağırlıklarını ayarlıyoruz.
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Checkpoint yüklendi: {ckpt_path}")
        except Exception as e:
            print(f"Checkpoint yüklenemedi: {e}. Lütfen modelinizi eğitin.")
    else:
        print("Checkpoint bulunamadı, lütfen modeli eğitin.")
        return

    test_dataset = ImageDatasetTest(image_folder="./dataset/test", sequence_length=args.seq_len)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for idx, (input_frames, target_frame) in enumerate(test_loader):
            input_frames = input_frames.to(device)  # Boyut: (batch, seq_len, channels, H, W)
            # Modelin ileri geçişi: Giriş verisine karşılık reconstruction üretir.
            reconstruction = model(input_frames)  # Çıktı boyutu da (batch, seq_len, channels, H, W)
            
            # Üretilen sonuçları kaydetmek için:
            save_image(input_frames[0], f"./results/input_{idx}.png", nrow=args.seq_len)
            save_image(reconstruction[0], f"./results/reconstruction_{idx}.png", nrow=args.seq_len)
            save_image(target_frame[0], f"./results/target_{idx}.png")
            
            print(f"Örnek {idx} kaydedildi.")
            
            if idx >= args.num_samples - 1:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--model', type=str, default='convlstm')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=5, help='Giriş sekansındaki frame sayısı')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps'], help='Cihaz: cpu veya mps')
    parser.add_argument('--num_samples', type=int, default=5, help='Testte kaç örnek görmek istersiniz')
    args = parser.parse_args()

    main(args)