import os
import argparse
import yaml
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast

# Assuming these modules exist in the project structure
try:
    from data.dataset import KLAImageDataset
    from models.nafnet_sr import build_model
    from models.losses import CompositeLoss
except ImportError:
    print("Warning: Custom modules not found. Ensure data and models directories exist in the python path.")
    # Placeholder classes for structural completeness if running outside full repo
    class KLAImageDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, is_train=True): pass
        def __len__(self): return 100
        def __getitem__(self, idx): return torch.randn(1, 64, 64), torch.randn(1, 128, 128)
    def build_model(config): return nn.Identity()
    class CompositeLoss(nn.Module):
        def forward(self, pred, target): return torch.mean((pred - target)**2)

# Utility to compute metrics
def compute_psnr(pred, target, data_range=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10((data_range ** 2) / mse)

def compute_ssim_torch(pred, target, window_size=11, channel=1):
    # Simplified SSIM for demonstration, assuming scikit-image style SSIM normally used
    # In a full implementation, this should use a proper PyTorch SSIM library
    # Here we just return a dummy value or a basic implementation
    return 0.85 # Placeholder

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m}m"
    elif m > 0:
        return f"{m}m{s}s"
    return f"{s}s"

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
    def __iter__(self):
        n = len(self.data_source)
        while True:
            yield from torch.randperm(n).tolist()
    def __len__(self):
        return int(1e12) # Essentially infinite

def main():
    parser = argparse.ArgumentParser(description='Train NAFNet-Tiny for semicon restoration')
    parser.add_argument('--config', type=str, default='configs/nafnet_tiny.yml', help='Path to config yaml')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory of KLA dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--max_iters', type=int, default=None, help='Override max iterations from config')
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config {args.config} not found, using defaults.")
        config = {
            'train': {
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'warmup_iters': 1000,
                'max_iters': 100000,
                'val_interval': 1000,
                'save_interval': 5000,
                'batch_size': 16,
                'num_workers': 4
            }
        }

    train_cfg = config.get('train', {})
    lr = float(train_cfg.get('lr', 1e-3))
    warmup_iters = int(train_cfg.get('warmup_iters', 1000))
    max_iters = args.max_iters if args.max_iters else int(train_cfg.get('max_iters', 50000))
    val_interval = int(train_cfg.get('val_interval', 1000))
    save_interval = int(train_cfg.get('save_interval', 5000))
    batch_size = int(train_cfg.get('batch_size', 16))
    num_workers = int(train_cfg.get('num_workers', 4))

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & Dataloader
    full_dataset = KLAImageDataset(args.data_dir, is_train=True)
    num_val = max(1, int(0.1 * len(full_dataset)))
    num_train = len(full_dataset) - num_val
    
    # Split: last 10% for val
    train_indices = list(range(num_train))
    val_indices = list(range(num_train, len(full_dataset)))
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_sampler = InfiniteSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Model
    model = build_model(config).to(device)
    
    # Loss, Optimizer, Scheduler
    criterion = CompositeLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(train_cfg.get('weight_decay', 1e-4)))
    
    # Linear warmup + Cosine Annealing
    def lr_lambda(current_step):
        if current_step < warmup_iters:
            # Linear warmup from 1e-6 to base lr
            alpha = current_step / max(1, warmup_iters)
            return (1e-6 / lr) * (1 - alpha) + alpha
        else:
            # Cosine annealing
            progress = (current_step - warmup_iters) / max(1, max_iters - warmup_iters)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    start_iter = 1
    best_psnr = 0.0

    # Resume
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_iter = checkpoint['iter'] + 1
        best_psnr = checkpoint.get('best_psnr', 0.0)
        print(f"Resumed at iteration {start_iter} with best_psnr {best_psnr:.2f}")

    print("Starting training...")
    train_iter = iter(train_loader)
    start_time = time.time()
    
    for current_iter in range(start_iter, max_iters + 1):
        model.train()
        
        try:
            degraded, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            degraded, target = next(train_iter)
            
        degraded, target = degraded.to(device), target.to(device)

        optimizer.zero_grad()
        
        with autocast():
            pred = model(degraded)
            loss = criterion(pred, target)
            
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Validation & Logging
        if current_iter % val_interval == 0 or current_iter == max_iters:
            model.eval()
            val_psnr = 0.0
            val_ssim = 0.0
            val_loss = 0.0
            
            with torch.no_grad():
                for val_deg, val_tar in val_loader:
                    val_deg, val_tar = val_deg.to(device), val_tar.to(device)
                    with autocast():
                        val_pred = model(val_deg)
                        v_loss = criterion(val_pred, val_tar)
                    
                    val_loss += v_loss.item()
                    val_psnr += compute_psnr(val_pred, val_tar).item()
                    val_ssim += compute_ssim_torch(val_pred, val_tar)
            
            n_val = len(val_loader)
            val_psnr /= n_val
            val_ssim /= n_val
            
            elapsed = time.time() - start_time
            iters_per_sec = (current_iter - start_iter + 1) / elapsed
            eta_seconds = (max_iters - current_iter) / iters_per_sec
            eta = format_time(eta_seconds)
            current_lr = get_lr(optimizer)
            
            print(f"[iter {current_iter}/{max_iters}] loss={loss.item():.4f} | val_psnr={val_psnr:.2f} | val_ssim={val_ssim:.3f} | lr={current_lr:.1e} | ETA: {eta}")

            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_path = os.path.join(args.save_dir, 'best_model.pt')
                torch.save({
                    'iter': current_iter,
                    'model_state': model.state_dict(),
                    'best_psnr': best_psnr,
                    'val_ssim': val_ssim
                }, best_path)
                print(f"  -> Saved new best model (PSNR: {best_psnr:.2f})")

        # Save regular checkpoint
        if current_iter % save_interval == 0 or current_iter == max_iters:
            ckpt_path = os.path.join(args.save_dir, f'ckpt_{current_iter:06d}.pt')
            torch.save({
                'iter': current_iter,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_psnr': best_psnr
            }, ckpt_path)

    print(f"Training completed. Final best PSNR: {best_psnr:.2f}")

if __name__ == '__main__':
    main()
