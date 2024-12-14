import os
import numpy as np
from PIL import Image

def save_frames_as_npy(input_dir, output_dir, img_size=(64, 64)):
    os.makedirs(output_dir, exist_ok=True)
    frames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')],
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

    video_array = []
    for frame in frames:
        img = Image.open(frame).convert('L')
        img = img.resize(img_size)
        video_array.append(np.array(img, dtype=np.uint8))

    video_array = np.stack(video_array, axis=0)
    npy_path = os.path.join(output_dir, "data.npy")
    np.save(npy_path, video_array)
    print(f"Saved: {npy_path}")
