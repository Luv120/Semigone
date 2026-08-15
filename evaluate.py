import os
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
    from skimage.metrics import structural_similarity as ssim_metric
except ImportError:
    psnr_metric = None

try:
    from models.nafnet_sr import build_model
except ImportError:
    # Dummy build model for standalone capability if module not found
    def build_model(config=None):
        return torch.nn.Identity()

def pad_tensor(x, mod=16):
    _, _, h, w = x.size()
    pad_h = (mod - h % mod) % mod
    pad_w = (mod - w % mod) % mod
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, pad_h, pad_w

def unpad_tensor(x, pad_h, pad_w, scale=1):
    pad_h = pad_h * scale
    pad_w = pad_w * scale
    _, _, h, w = x.size()
    return x[:, :, :h-pad_h, :w-pad_w]

def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        img = np.load(path)
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
    else:
        img = Image.open(path).convert('L')
        img = np.array(img, dtype=np.float32) / 255.0
    return img

def save_image(img_np, path, ref_path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(ref_path)[1].lower()
    
    if ext == '.npy':
        np.save(path, img_np)
    else:
        img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(path)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Restored Images')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory of degraded test images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save restored images')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights (.pt)')
    parser.add_argument('--config', type=str, default=None, help='Path to config yaml')
    parser.add_argument('--gt_dir', type=str, default=None, help='Optional ground truth directory for metrics')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    scale = 1
    model = None
    use_dummy = False
    
    if args.weights and os.path.exists(args.weights):
        # In a real scenario, scale would be determined from config/model
        # We assume scale=2 if 'sr' or '2x' is in config/weights name, otherwise 1
        if args.config and '2x' in args.config:
            scale = 2
        
        try:
            model = build_model(args.config)
            checkpoint = torch.load(args.weights, map_location=device)
            state_dict = checkpoint.get('model_state', checkpoint)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print(f"Loaded weights from {args.weights} (scale={scale})")
        except Exception as e:
            print(f"Could not load model: {e}. Falling back to dummy baseline.")
            use_dummy = True
    else:
        print("No valid weights provided. Using bicubic upsampling as dummy baseline.")
        use_dummy = True
        scale = 2 # Assuming super-resolution baseline if nothing specified

    # Find all images
    valid_exts = ('.npy', '.png', '.tif', '.tiff')
    input_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                input_files.append(os.path.join(root, f))
                
    input_files = sorted(input_files)
    total_files = len(input_files)
    print(f"Found {total_files} files to process.")

    psnr_list, ssim_list = [], []

    for idx, in_path in enumerate(input_files):
        # Preserve directory structure
        rel_path = os.path.relpath(in_path, args.input_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        
        print(f"[{idx+1}/{total_files}] Processing {os.path.basename(in_path)} -> {out_path}")
        
        # Load and process
        img_np = load_image(in_path)
        
        # Convert to tensor
        img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
        
        if use_dummy:
            # Dummy baseline (Bicubic)
            if scale > 1:
                out_t = F.interpolate(img_t, scale_factor=scale, mode='bicubic', align_corners=False)
            else:
                out_t = img_t
        else:
            # NAFNet processing
            img_t_padded, pad_h, pad_w = pad_tensor(img_t, mod=16)
            with torch.no_grad():
                # Some AMP usage if desired
                out_t = model(img_t_padded)
            out_t = unpad_tensor(out_t, pad_h, pad_w, scale=scale)
            
        out_np = out_t.squeeze().cpu().numpy()
        out_np = np.clip(out_np, 0.0, 1.0)
        
        save_image(out_np, out_path, in_path)
        
        # Calculate metrics if GT provided
        if args.gt_dir and psnr_metric is not None:
            gt_path = os.path.join(args.gt_dir, rel_path)
            if os.path.exists(gt_path):
                gt_np = load_image(gt_path)
                p = psnr_metric(gt_np, out_np, data_range=1.0)
                s = ssim_metric(gt_np, out_np, data_range=1.0, win_size=11)
                psnr_list.append(p)
                ssim_list.append(s)

    if args.gt_dir and psnr_list:
        print(f"\n--- Evaluation Results ---")
        print(f"Average PSNR: {np.mean(psnr_list):.2f} dB")
        print(f"Average SSIM: {np.mean(ssim_list):.4f}")

if __name__ == '__main__':
    main()
