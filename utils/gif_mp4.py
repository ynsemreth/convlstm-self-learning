import imageio
import cv2
import numpy as np
import torch
import os

def save_all_to_gif(test_loader, model, device, output_path="all_results.gif", frame_duration=0.1):
    model.eval()
    input_frames, target_frames, output_frames = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            inputs_np = (inputs.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            targets_np = (targets.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            outputs_np = (outputs.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            if inputs_np.ndim == 5:
                b, t, c, h, w = inputs_np.shape
                inputs_np = inputs_np.reshape(b*t, c, h, w) 
                targets_np = targets_np.reshape(b*t, c, h, w)
                outputs_np = outputs_np.reshape(b*t, c, h, w)


            if inputs_np.ndim == 4:
                inputs_np = np.transpose(inputs_np, (0, 2, 3, 1))
                targets_np = np.transpose(targets_np, (0, 2, 3, 1))
                outputs_np = np.transpose(outputs_np, (0, 2, 3, 1))

            if inputs_np.shape[-1] == 1:
                inputs_np = np.repeat(inputs_np, 3, axis=-1)
            if targets_np.shape[-1] == 1:
                targets_np = np.repeat(targets_np, 3, axis=-1)
            if outputs_np.shape[-1] == 1:
                outputs_np = np.repeat(outputs_np, 3, axis=-1)

            input_frames.extend(list(inputs_np))
            target_frames.extend(list(targets_np))
            output_frames.extend(list(outputs_np))

    with imageio.get_writer(output_path, mode="I", duration=frame_duration) as writer:
        for i in range(len(input_frames)):
            # Boyutları eşitle
            H, W, _ = input_frames[i].shape

            if target_frames[i].shape[:2] != (H, W):
                target_frames[i] = cv2.resize(target_frames[i], (W, H))
            if output_frames[i].shape[:2] != (H, W):
                output_frames[i] = cv2.resize(output_frames[i], (W, H))

            combined_frame = np.concatenate([input_frames[i],
                                             target_frames[i],
                                             output_frames[i]], axis=1)
            writer.append_data(combined_frame)

    print(f"GIF kaydedildi: {output_path}")


def gif_to_video_with_opencv(gif_path, video_path, fps=10):
    gif = imageio.mimread(gif_path)
    height, width, _ = gif[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in gif:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"GIF MP4 formatına dönüştürüldü: {video_path}")

    if os.path.exists(gif_path):
        os.remove(gif_path)
        print(f"GIF dosyası silindi: {gif_path}")
    else:
        print(f"GIF dosyası bulunamadı: {gif_path}")