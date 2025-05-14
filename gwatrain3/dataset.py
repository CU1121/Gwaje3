
import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

class ConditionalLowLightDataset(Dataset):
    def __init__(self, low_dir, enh_dir, meta_file, transform=None, augment=False):
        self.low_dir = low_dir
        self.enh_dir = enh_dir
        self.low_files = sorted(os.listdir(low_dir))
        self.enh_files = sorted([
            f for f in os.listdir(enh_dir)
            if not f.startswith('mask_') and f.lower().endswith(('.jpg', '.png'))
        ])
        with open(meta_file) as f:
            self.meta = json.load(f)
        self.transform = transform
        self.augment = augment
        self.aug = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        ])

    def __len__(self):
        return len(self.enh_files)

    def __getitem__(self, idx):
        enh = self.enh_files[idx]
        base = enh.split('_')[0] + '.jpg'
        low_path = os.path.join(self.low_dir, base)
        enh_path = os.path.join(self.enh_dir, enh)
        mask_path = os.path.join(self.enh_dir, f"mask_{enh}")

        low = cv2.imread(low_path)
        enh_img = cv2.imread(enh_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if low is None or enh_img is None:
            raise FileNotFoundError(f"로딩 실패: {low_path} 또는 {enh_path}")

        low_rgb = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
        enh_rgb = cv2.cvtColor(enh_img, cv2.COLOR_BGR2RGB)
        if self.transform:
            low_t = self.transform(low_rgb)
            enh_t = self.transform(enh_rgb)
        else:
            low_t = torch.tensor(low_rgb).permute(2,0,1).float().div(255)
            enh_t = torch.tensor(enh_rgb).permute(2,0,1).float().div(255)
        if self.augment:
            low_t = self.aug(low_t)

        if mask is None:
            m = np.ones((256,256), dtype=np.float32)
        else:
            try:
                m = cv2.resize(mask, (256,256)).astype(np.float32) / 255.0
            except cv2.error:
                m = np.ones((256,256), dtype=np.float32)
        m_t = torch.tensor(m, dtype=torch.float32).unsqueeze(0)

        md = self.meta[enh]
        brightness = md['brightness'] / 255.0
        color_shifts = [c / 255.0 for c in md['color_shift']]
        cond = torch.tensor([brightness] + color_shifts, dtype=torch.float32)

        return low_t, enh_t, cond, m_t
