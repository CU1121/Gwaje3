import os
import cv2
import numpy as np
import json
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)
    ])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.T
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True):
    (_, channel, height, width) = img1.size()
    if window is None:
        window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map

def analyze_and_generate_metadata(lowlight_dir, enhanced_dir, save_path="metadata.json"):
    metadata = {}
    lowlight_images = sorted(os.listdir(lowlight_dir))
    enhanced_images = sorted([f for f in os.listdir(enhanced_dir) if not f.startswith("mask_") and f.endswith(('.jpg', '.png')) and f.startswith("img")])

    for low_file, enh_file in zip(lowlight_images, enhanced_images):
        low_img = cv2.imread(os.path.join(lowlight_dir, low_file))
        enh_img = cv2.imread(os.path.join(enhanced_dir, enh_file))

        low_img_rgb = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        enh_img_rgb = cv2.cvtColor(enh_img, cv2.COLOR_BGR2RGB).astype(np.float32)

        diff = cv2.absdiff(low_img_rgb, enh_img_rgb)
        diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        change_mask = (diff_gray > 15).astype(np.uint8) * 255

        if np.sum(change_mask) > 0:
            changed_pixels_low = low_img_rgb[change_mask > 0]
            changed_pixels_enh = enh_img_rgb[change_mask > 0]

            brightness_diff = (np.mean(cv2.cvtColor(changed_pixels_enh.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2GRAY)) -
                               np.mean(cv2.cvtColor(changed_pixels_low.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2GRAY)))
            color_diff = np.mean(changed_pixels_enh - changed_pixels_low, axis=0)
        else:
            brightness_diff = 0.0
            color_diff = [0.0, 0.0, 0.0]

        metadata[enh_file] = {
            "brightness": brightness_diff,
            "color_shift": color_diff.tolist()
        }

    with open(os.path.join(enhanced_dir, save_path), "w") as f:
        json.dump(metadata, f, indent=4)

class ConditionalLowLightDataset(Dataset):
    def __init__(self, lowlight_dir, enhanced_dir, metadata_file, transform=None):
        self.lowlight_dir = lowlight_dir
        self.enhanced_dir = enhanced_dir
        self.lowlight_images = sorted(os.listdir(lowlight_dir))
        self.enhanced_images = sorted([f for f in os.listdir(enhanced_dir) if not f.startswith("mask_") and f.startswith("img")])
        self.transform = transform

        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.enhanced_images)

    def __getitem__(self, idx):
        enh_img_name = self.enhanced_images[idx]
        base_name = enh_img_name.split('_')[0]
        low_img_name = base_name + ".jpg"

        low_img_path = os.path.join(self.lowlight_dir, low_img_name)
        enh_img_path = os.path.join(self.enhanced_dir, enh_img_name)

        if not os.path.exists(low_img_path):
            raise FileNotFoundError(f"원본 이미지 없음: {low_img_path}")

        low_img = cv2.imread(low_img_path)
        enh_img = cv2.imread(enh_img_path)

        if low_img is None or enh_img is None:
            raise ValueError(f"OpenCV 이미지 로딩 실패: {low_img_path} 또는 {enh_img_path}")

        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        enh_img = cv2.cvtColor(enh_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            low_tensor = self.transform(low_img)
            enh_tensor = self.transform(enh_img)

        info = self.metadata[enh_img_name]
        condition_vector = torch.tensor([info["brightness"]] + info["color_shift"], dtype=torch.float32)
        return low_tensor, enh_tensor, condition_vector

class UNetConditionalModel(nn.Module):
    def __init__(self, condition_dim=4):
        super(UNetConditionalModel, self).__init__()
        self.condition_fc = nn.Linear(condition_dim, 256 * 256)

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU()
            )

        self.enc1 = conv_block(4, 64)
        self.enc2 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = conv_block(256 + 128, 128)
        self.dec1 = conv_block(128 + 64, 64)

        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x, condition):
        b = x.size(0)
        cond_map = self.condition_fc(condition).view(b, 1, 256, 256)
        x = torch.cat([x, cond_map], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bn = self.bottleneck(self.pool(e2))

        d2 = self.dec2(torch.cat([self.up(bn), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        out = self.final(d1)
        return out

def safe_save_model(model, path="final_conditional_model.pth"):
    temp_path = path + ".temp"
    try:
        torch.save(model.state_dict(), temp_path)
        os.replace(temp_path, path)
        print("✅ 모델이 안전하게 저장되었습니다")
    except:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print("❌ 저장 중 오류 발생")

def train_conditional_model(lowlight_dir, enhanced_dir, metadata_file, epochs=4, batch_size=4, lr=0.001):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    dataset = ConditionalLowLightDataset(lowlight_dir, enhanced_dir, metadata_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNetConditionalModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for low, enh, cond in dataloader:
            low, enh, cond = low.to(device), enh.to(device), cond.to(device)
            optimizer.zero_grad()
            out = model(low, cond)
            loss = mse(out, enh) + (1 - ssim(out, enh)) * 0.1
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    safe_save_model(model, "final_conditional_model.pth")

if __name__ == "__main__":
    lowlight_folder = input("원본 이미지 폴더 경로: ")
    enhanced_folder = input("보정 이미지 폴더 경로: ")
    analyze_and_generate_metadata(lowlight_folder, enhanced_folder)
    train_conditional_model(lowlight_folder, enhanced_folder, os.path.join(enhanced_folder, "metadata.json"))
