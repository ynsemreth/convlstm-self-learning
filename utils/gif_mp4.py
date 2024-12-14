import imageio
import cv2
import numpy as np
import torch

def save_all_to_gif(test_loader, model, device, output_path="all_results.gif", frame_duration=0.1):
    model.eval()
    input_frames, target_frames, output_frames = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            input_frames.extend(((inputs.cpu().numpy() * 255).clip(0, 255)).astype(np.uint8).squeeze(2))
            target_frames.extend(((targets.cpu().numpy() * 255).clip(0, 255)).astype(np.uint8).squeeze(2))
            output_frames.extend(((outputs.cpu().numpy() * 255).clip(0, 255)).astype(np.uint8).squeeze(2))

    with imageio.get_writer(output_path, mode="I", duration=frame_duration) as writer:
        for i in range(len(input_frames)):
            combined_frame = np.concatenate(
                [input_frames[i], target_frames[i], output_frames[i]], axis=1
            )
            writer.append_data(combined_frame)

    print(f"Saved combined GIF: {output_path}")

import imageio
import cv2
import os

def gif_to_video_with_opencv(gif_path, video_path, fps=10):
    gif = imageio.mimread(gif_path)
    height, width, _ = gif[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in gif:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"Converted GIF to video: {video_path}")

    if os.path.exists(gif_path):
        os.remove(gif_path)
        print(f"Deleted GIF file: {gif_path}")
    else:
        print(f"GIF file not found: {gif_path}")
