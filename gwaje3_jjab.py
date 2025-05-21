import os
import cv2
import numpy as np
import json
import torch
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
import torch.nn.functional as F
import kornia.color as KC  # HSV ↔ RGB 변환
import lpips                 # LPIPS 지각 손실
from kornia.filters import Sobel  # Sobel 엣지 필터

# ====================================================
# 글로벌 이미지 크기 설정 (H, W)
# ====================================================
IMG_H = 400
IMG_W = 600

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================================================
# 유틸리티 함수
# ====================================================
def make_laplacian_kernel(k: int):
    lap = -torch.ones((k, k), dtype=torch.float32)
    lap[k//2, k//2] = k * k - 1
    return lap

def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0):
    mse = F.mse_loss(x, y, reduction='none').flatten(start_dim=1).mean(dim=1)
    return (10 * torch.log10(max_val**2 / (mse + 1e-8))).mean().item()

# ====================================================
# 모델 블록 정의
# ====================================================
class SimpleEdgeExtractor(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch//r, 1), nn.ReLU(),
            nn.Conv2d(ch//r, ch, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.net(x)

class UNetConditionalModel(nn.Module):
    def __init__(self, cond_dim: int = 5):
        super().__init__()
        self.cond_fc = nn.Linear(cond_dim, IMG_H * IMG_W)
        def blk(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(),
                nn.Conv2d(oc, oc, 3, padding=1), nn.ReLU(),
                SEBlock(oc)
            )
        self.enc1 = blk(4, 64)
        self.enc2 = blk(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bott = blk(128, 256)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = blk(256 + 128 + 2, 128)
        self.dec1 = blk(128 + 64 + 2, 64)
        self.final = nn.Conv2d(64, 3, 1)
    def forward(self, x, cond, struct):
        b = x.size(0)
        cm = self.cond_fc(cond).view(b, 1, IMG_H, IMG_W)
        x = torch.cat([x, cm], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bn = self.bott(self.pool(e2))
        s_down = F.interpolate(struct, scale_factor=0.5, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([self.up(bn), e2, s_down], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1, struct], dim=1))
        return self.final(d1)

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights).features[:9].eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.net = vgg.to(device)
        self.crit = nn.MSELoss()
    def forward(self, x, y):
        return self.crit(self.net(x), self.net(y))

# ====================================================
# 전역 forward_pass 정의 (train/infer 공용)
# ====================================================
def forward_pass(lo, cond, msk, model, edge_net, sobel):
    b = cond[:, 0:1]; cs = cond[:, 1:4]; s = cond[:, 4:5]
    lo_h = KC.rgb_to_hsv(lo)
    lo_h[:, 2:3, :, :] = torch.clamp(lo_h[:, 2:3, :, :] + b.view(-1, 1, 1, 1) * msk, 0, 1)
    lo_b = KC.hsv_to_rgb(lo_h)
    lo_bc = torch.clamp(lo_b + cs.view(-1, 3, 1, 1) * msk, 0, 1)
    gray = KC.rgb_to_grayscale(lo_bc)
    sm = torch.norm(sobel(gray), dim=1, keepdim=True)
    lm = edge_net(lo_bc)
    struct = torch.cat([sm, lm], dim=1)
    res = model(lo_bc, cond, struct)
    out_raw = torch.clamp(lo_bc + res, 0, 1)
    out_h = KC.rgb_to_hsv(out_raw)
    out_h[:, 1:2, :, :] = torch.clamp(out_h[:, 1:2, :, :] + s.view(-1, 1, 1, 1) * msk, 0, 1)
    return KC.hsv_to_rgb(out_h)

# ====================================================
# 1. 메타데이터 생성
# ====================================================
def analyze_and_generate_metadata(low_dir, enh_dir, save_name="metadata.json"):
    meta = {}
    low_fs = sorted(os.listdir(low_dir))
    enh_fs = sorted([f for f in os.listdir(enh_dir)
                     if not f.startswith('mask_') and f.lower().endswith(('.jpg', '.png'))])
    for lf, ef in zip(low_fs, enh_fs):
        low_bgr = cv2.imread(os.path.join(low_dir, lf))
        enh_bgr = cv2.imread(os.path.join(enh_dir, ef))
        if low_bgr is None or enh_bgr is None: continue
        low_hsv = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        enh_hsv = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        diff = cv2.absdiff(low_bgr, enh_bgr)
        mask = (cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) > 15).astype(np.uint8) * 255
        idx = mask > 0
        v_diff = float(np.mean(enh_hsv[..., 2][idx]) - np.mean(low_hsv[..., 2][idx])) if idx.any() else 0.0
        s_diff = float(np.mean(enh_hsv[..., 1][idx]) - np.mean(low_hsv[..., 1][idx])) if idx.any() else 0.0
        tmp = low_hsv.copy(); tmp[..., 2] = np.clip(tmp[..., 2] + v_diff, 0, 255)
        adj_bgr = cv2.cvtColor(tmp.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        rgb_diff = np.mean(enh_bgr.astype(np.float32) - adj_bgr, axis=(0, 1)).tolist()
        meta[ef] = {"brightness": v_diff, "color_shift": rgb_diff, "sat_shift": s_diff}
        cv2.imwrite(os.path.join(enh_dir, f"mask_{ef}"), mask)
    with open(os.path.join(enh_dir, save_name), 'w') as f:
        json.dump(meta, f, indent=4)
    print("✅ 메타데이터 생성 완료.")

# ====================================================
# 2. 데이터셋 정의
# ====================================================
class ConditionalLowLightDataset(Dataset):
    def __init__(self, low_dir, enh_dir, meta_file, transform=None, augment=False):
        self.low_dir = low_dir; self.enh_dir = enh_dir
        self.enh_fs = sorted([f for f in os.listdir(enh_dir)
                               if not f.startswith('mask_')])
        with open(meta_file) as f: self.meta = json.load(f)
        self.transform = transform; self.augment = augment
        self.aug = T.ColorJitter(0.3, 0.3, 0.3, 0.1)
    def __len__(self): return len(self.enh_fs)
    def __getitem__(self, i):
        ef = self.enh_fs[i]; lf = ef.split('_')[0] + '.jpg'
        low_bgr = cv2.imread(os.path.join(self.low_dir, lf))
        enh_bgr = cv2.imread(os.path.join(self.enh_dir, ef))
        mask = cv2.imread(os.path.join(self.enh_dir, f"mask_{ef}"), cv2.IMREAD_GRAYSCALE)
        low = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2RGB); enh = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB)
        to_t = self.transform if self.transform else (lambda x: torch.tensor(x).permute(2, 0, 1).float() / 255)
        low_t, enh_t = to_t(low), to_t(enh)
        if self.augment: low_t = self.aug(low_t)
        m = np.ones((IMG_H, IMG_W), np.float32) if mask is None else cv2.resize(mask, (IMG_W, IMG_H)).astype(np.float32) / 255.
        m_t = torch.tensor(m).unsqueeze(0)
        md = self.meta[ef]
        cond = torch.tensor([
            md['brightness'] / 255.0,
            *[c / 255.0 for c in md['color_shift']],
            md['sat_shift'] / 255.0
        ], dtype=torch.float32)
        return low_t, enh_t, cond, m_t

# ====================================================
# 3. 학습 루프
# ====================================================
def train(low_dir, enh_dir, meta_file, epochs=100, bs=4, lr=2e-3):
    ds = ConditionalLowLightDataset(
        low_dir, enh_dir, meta_file,
        transform=T.Compose([T.ToPILImage(), T.Resize((IMG_H, IMG_W)), T.ToTensor()]),
        augment=True
    )
    n_val = int(0.2 * len(ds)); tr, va = random_split(ds, [len(ds) - n_val, n_val])
    tr_dl = DataLoader(tr, bs, shuffle=True); va_dl = DataLoader(va, bs)
    model = UNetConditionalModel().to(device)
    edge_net = SimpleEdgeExtractor().to(device)
    sobel = Sobel().to(device)
    opt = optim.Adam(model.parameters(), lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt)
    mse = nn.MSELoss()
    perc = VGGPerceptualLoss().to(device)
    lp = lpips.LPIPS(net='vgg').to(device)
    best, pat = float('inf'), 0

    for e in range(epochs):
        model.train(); tr_loss = 0
        for lo, eh, cond, msk in tr_dl:
            lo, eh, cond, msk = lo.to(device), eh.to(device), cond.to(device), msk.to(device)
            msk = msk.unsqueeze(1)
            opt.zero_grad()
            out = forward_pass(lo, cond, msk, model, edge_net, sobel)
            loss = mse(out, eh) + perc(out, eh) + lp(out, eh).mean()
            loss.backward(); opt.step()
            tr_loss += loss.item()
        model.eval(); val_loss = 0
        with torch.no_grad():
            for lo, eh, cond, msk in va_dl:
                lo, eh, cond, msk = lo.to(device), eh.to(device), cond.to(device), msk.to(device)
                msk = msk.unsqueeze(1)
                out = forward_pass(lo, cond, msk, model, edge_net, sobel)
                val_loss += (mse(out, eh) + perc(out, eh) + lp(out, eh).mean()).item()
        val_loss /= len(va_dl)
        print(f"Epoch {e+1}/{epochs} → Train: {tr_loss/len(tr_dl):.4f}, Val: {val_loss:.4f}")
        sched.step(val_loss)
        if val_loss < best:
            best = val_loss; pat = 0
        else:
            pat += 1
        if pat > 15:
            print("Early stopping triggered")
            break
    torch.save(model.state_dict(), 'final.pth')
    print("✅ 학습 완료")

# ====================================================
# 4. 추론
# ====================================================
def inference(path, brightness, rgb_shift, sat_shift):
    img = cv2.imread(path)
    transform = T.Compose([T.ToPILImage(), T.Resize((IMG_H, IMG_W)), T.ToTensor()])
    in_t = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    cond = torch.tensor([[brightness/255.0, *[c/255.0 for c in rgb_shift], sat_shift/255.0]], dtype=torch.float32).to(device)
    msk = torch.ones(1, 1, IMG_H, IMG_W, device=device)
    model = UNetConditionalModel().to(device)
    model.load_state_dict(torch.load('final.pth', map_location=device))
    model.eval()
    edge_net = SimpleEdgeExtractor().to(device)
    sobel = Sobel().to(device)
    with torch.no_grad():
        out = forward_pass(in_t, cond, msk, model, edge_net, sobel)
    res = (out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    cv2.imshow('AI 보정 결과', cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    mode = input('Mode(train/infer): ')
    if mode == 'train':
        low = input('Low dir: ')
        enh = input('Enh dir: ')
        analyze_and_generate_metadata(low, enh)
        train(low, enh, os.path.join(enh, 'metadata.json'))
    elif mode == 'infer':
        path = input('Image path: ')
        v = float(input('ΔV (brightness adjustment): '))
        r = list(map(float, input('RGB shifts (R G B): ').split()))
        s = float(input('ΔS (saturation adjustment): '))
        inference(path, v, r, s)
    else:
        print('Unknown mode')
