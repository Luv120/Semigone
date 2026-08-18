import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
    from skimage.metrics import structural_similarity as ssim_metric
except ImportError:
    psnr_metric = None
    ssim_metric = None

# Ensure project root is on path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.nafnet_sr import build_model


def load_image(path):
    """Load grayscale image as float32 numpy array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        img = np.load(path).astype(np.float32)
    else:
        img = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0

    # Normalize: if values exceed 1.0 (speckle noise), divide by max
    if img.max() > 1.0:
        img = img / img.max()
    return img


def save_image(img_np, path, ref_path):
    """Save output in same format as input."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    ext = os.path.splitext(ref_path)[1].lower()

    if ext == '.npy':
        np.save(path, img_np.astype(np.float32))
    else:
        img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(path)


def main():
    parser = argparse.ArgumentParser(description='Evaluate — Semiconductor Image Restoration (Team Semigone)')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory of degraded test images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save restored images')
    parser.add_argument('--weights', type=str, default='weights/nafnet_tiny.pt', help='Path to model weights (.pt)')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config yaml')
    parser.add_argument('--gt_dir', type=str, default=None, help='Optional ground truth directory for metrics')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # ---- Load config ----
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Sensible defaults matching NAFNet-Tiny
        config = {
            'model': {
                'img_channel': 1, 'width': 16,
                'enc_blk_nums': [1, 1, 1, 2], 'middle_blk_num': 1,
                'dec_blk_nums': [1, 1, 1, 1], 'scale': 2
            }
        }

    scale = config.get('model', {}).get('scale', 2)

    # ---- Load model ----
    model = build_model(config)
    model.to(device)
    model.eval()

    weights_path = args.weights
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        # Handle both full checkpoint dicts and raw state_dicts
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path} (scale={scale})")
    else:
        print(f"WARNING: Weights file '{weights_path}' not found. Running with random weights.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: NAFNet-Tiny | {n_params:,} params | scale={scale}x")

    # ---- Find input files ----
    valid_exts = ('.npy', '.png', '.tif', '.tiff', '.jpg', '.jpeg')
    input_files = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(args.input_dir)
        for f in files if f.lower().endswith(valid_exts)
    ])
    total_files = len(input_files)
    print(f"Found {total_files} files to process.")

    if total_files == 0:
        print("ERROR: No input files found. Check --input_dir path.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    psnr_list, ssim_list = [], []
    times = []

    for idx, in_path in enumerate(input_files):
        rel_path = os.path.relpath(in_path, args.input_dir)
        out_path = os.path.join(args.output_dir, rel_path)

        # Load and normalize
        img_np = load_image(in_path)

        # To tensor: [1, 1, H, W]
        img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(device)

        # Run model
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()

        with torch.no_grad():
            out_t = model(img_t)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

        out_np = out_t.squeeze().cpu().numpy()
        out_np = np.clip(out_np, 0.0, 1.0)

        save_image(out_np, out_path, in_path)

        if (idx + 1) % 50 == 0 or (idx + 1) == total_files:
            print(f"[{idx+1}/{total_files}] {os.path.basename(in_path)} -> {out_np.shape} ({elapsed_ms:.1f}ms)")

        # Calculate metrics if GT provided
        if args.gt_dir and psnr_metric is not None:
            gt_path = os.path.join(args.gt_dir, rel_path)
            if os.path.exists(gt_path):
                gt_np = load_image(gt_path)
                # Ensure same shape
                if gt_np.shape == out_np.shape:
                    p = psnr_metric(gt_np, out_np, data_range=1.0)
                    s = ssim_metric(gt_np, out_np, data_range=1.0)
                    psnr_list.append(p)
                    ssim_list.append(s)
                else:
                    print(f"  Shape mismatch: output {out_np.shape} vs GT {gt_np.shape}, skipping metrics")

    # ---- Summary ----
    print(f"\n{'='*50}")
    print(f"Processed {total_files} images")
    print(f"Avg inference: {np.mean(times):.2f} ms | {1000/np.mean(times):.0f} FPS")
    print(f"Output dir: {args.output_dir}")

    if psnr_list:
        print(f"\n--- Metrics (vs Ground Truth) ---")
        print(f"Average PSNR: {np.mean(psnr_list):.2f} dB")
        print(f"Average SSIM: {np.mean(ssim_list):.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
