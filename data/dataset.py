import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as TF

class PairedDataset(Dataset):
    def __init__(self, input_dir, target_dir, lr_patch_size=64, is_train=True):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.lr_patch_size = lr_patch_size
        self.is_train = is_train
        
        self.input_paths = sorted(glob.glob(os.path.join(input_dir, '*')))
        self.target_paths = sorted(glob.glob(os.path.join(target_dir, '*')))
        
        assert len(self.input_paths) == len(self.target_paths) and len(self.input_paths) > 0

    def __len__(self):
        return len(self.input_paths)

    def load_img(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            img = np.load(path).astype(np.float32)
        else:
            img = np.array(Image.open(path)).astype(np.float32)
            
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        return img

    def __getitem__(self, idx):
        lr_path = self.input_paths[idx]
        hr_path = self.target_paths[idx]
        
        lr_img = self.load_img(lr_path)
        hr_img = self.load_img(hr_path)
        
        # normalization
        lr_max = lr_img.max()
        if lr_max > 0:
            lr_img = lr_img / lr_max
            
        hr_max = hr_img.max()
        if hr_max > 1.0: # uint8
            hr_img = hr_img / 255.0
            
        h_lr, w_lr, _ = lr_img.shape
        h_hr, w_hr, _ = hr_img.shape
        
        scale = h_hr // h_lr
        
        if self.is_train:
            # random crop
            h_lr_p = self.lr_patch_size
            w_lr_p = self.lr_patch_size
            
            h_hr_p = h_lr_p * scale
            w_hr_p = w_lr_p * scale
            
            x = random.randint(0, h_lr - h_lr_p)
            y = random.randint(0, w_lr - w_lr_p)
            
            lr_img = lr_img[x:x+h_lr_p, y:y+w_lr_p, :]
            hr_img = hr_img[x*scale:x*scale+h_hr_p, y*scale:y*scale+w_hr_p, :]
            
            # augmentations
            lr_tensor = torch.from_numpy(lr_img.transpose(2,0,1).copy())
            hr_tensor = torch.from_numpy(hr_img.transpose(2,0,1).copy())
            
            if random.random() < 0.5:
                lr_tensor = TF.hflip(lr_tensor)
                hr_tensor = TF.hflip(hr_tensor)
            if random.random() < 0.5:
                lr_tensor = TF.vflip(lr_tensor)
                hr_tensor = TF.vflip(hr_tensor)
            if random.random() < 0.5:
                rot = random.choice([90, 180, 270])
                lr_tensor = TF.rotate(lr_tensor, rot)
                hr_tensor = TF.rotate(hr_tensor, rot)
        else:
            lr_tensor = torch.from_numpy(lr_img.transpose(2,0,1).copy())
            hr_tensor = torch.from_numpy(hr_img.transpose(2,0,1).copy())
            
        return lr_tensor, hr_tensor


class TestDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.input_paths = sorted(glob.glob(os.path.join(input_dir, '*')))
        
    def __len__(self):
        return len(self.input_paths)
        
    def load_img(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            img = np.load(path).astype(np.float32)
        else:
            img = np.array(Image.open(path)).astype(np.float32)
            
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        return img
        
    def __getitem__(self, idx):
        lr_path = self.input_paths[idx]
        lr_img = self.load_img(lr_path)
        
        lr_max = lr_img.max()
        if lr_max > 0:
            lr_img = lr_img / lr_max
            
        lr_tensor = torch.from_numpy(lr_img.transpose(2,0,1).copy())
        return lr_tensor, os.path.basename(lr_path)
