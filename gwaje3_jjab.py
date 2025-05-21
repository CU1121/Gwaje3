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
import kornia.color as KC  # HSV·RGB 변환
import lpips                 # LPIPS 지각 손실
from kornia.filters import Sobel  # Sobel 엣지 필터

# ====================================================
# 글로벌 이미지 크기 설정 (H, W)
# ====================================================
IMG_H = 400  # height
IMG_W = 600  # width

# ====================================================
# 유틸리티
# ====================================================

def make_laplacian_kernel(k:int):
    lap = -torch.ones((k,k), dtype=torch.float32)
    lap[k//2, k//2] = k*k - 1
    return lap

def psnr(x:torch.Tensor, y:torch.Tensor, max_val:float=1.0):
    mse = F.mse_loss(x, y, reduction='none').flatten(start_dim=1).mean(dim=1)
    return (10 * torch.log10(max_val**2 / (mse + 1e-8))).mean().item()

# ====================================================
# 네트워크 블록
# ====================================================
class SimpleEdgeExtractor(nn.Module):
    def __init__(self, in_ch:int=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class SEBlock(nn.Module):
    def __init__(self, ch:int, r:int=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch//r, 1), nn.ReLU(),
            nn.Conv2d(ch//r, ch, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.net(x)

class UNetConditionalModel(nn.Module):
    """U‑Net + SE attention + 조건 임베딩."""
    def __init__(self, cond_dim:int=5):  # cond: brightness + RGB shift + S shift
        super().__init__()
        self.cond_fc = nn.Linear(cond_dim, IMG_H*IMG_W)
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
        self.dec2 = blk(256+128+2, 128)
        self.dec1 = blk(128+64+2, 64)
        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x, cond, struct_map):
        b = x.size(0)
        cm = self.cond_fc(cond).view(b, 1, IMG_H, IMG_W)
        x = torch.cat([x, cm], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bn = self.bott(self.pool(e2))
        s_down = F.interpolate(struct_map, scale_factor=0.5, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([self.up(bn), e2, s_down], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1, struct_map], dim=1))
        return self.final(d1)

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights).features[:9].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.net = vgg
        self.crit = nn.MSELoss()
    def forward(self, x, y):
        return self.crit(self.net(x), self.net(y))

# ====================================================
# 메타데이터 생성 (밝기·RGB·Saturation 차이)
# ====================================================

def analyze_and_generate_metadata(low_dir:str, enh_dir:str, save_name:str="metadata.json"):
    metadata = {}
    low_files = sorted(os.listdir(low_dir))
    enh_files = sorted([f for f in os.listdir(enh_dir)
                        if not f.startswith('mask_') and f.lower().endswith(('.jpg', '.png'))])
    for lf, ef in zip(low_files, enh_files):
        low_bgr = cv2.imread(os.path.join(low_dir, lf))
        enh_bgr = cv2.imread(os.path.join(enh_dir, ef))
        if low_bgr is None or enh_bgr is None:
            continue
        low_hsv, enh_hsv = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                            for img in (low_bgr, enh_bgr)]
        diff_rgb = cv2.absdiff(low_bgr, enh_bgr)
        mask = (cv2.cvtColor(diff_rgb, cv2.COLOR_BGR2GRAY) > 15).astype(np.uint8)*255
        idx = mask>0
        v_diff = float(np.mean(enh_hsv[...,2][idx]) - np.mean(low_hsv[...,2][idx])) if idx.any() else 0.0
        s_diff = float(np.mean(enh_hsv[...,1][idx]) - np.mean(low_hsv[...,1][idx])) if idx.any() else 0.0
        # 컬러 시프트 (RGB)
        lo_tmp = low_hsv.copy(); lo_tmp[...,2] = np.clip(lo_tmp[...,2] + v_diff, 0, 255)
        lo_bgr_adj = cv2.cvtColor(lo_tmp.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        color_diff = np.mean(enh_bgr.astype(np.float32) - lo_bgr_adj, axis=(0,1)).tolist()
        metadata[ef] = {
            "brightness": v_diff,
            "color_shift": color_diff,
            "sat_shift": s_diff
        }
        cv2.imwrite(os.path.join(enh_dir, f"mask_{ef}"), mask)
    with open(os.path.join(enh_dir, save_name), 'w') as f:
        json.dump(metadata, f, indent=4)
    print('✅ 메타데이터 생성 완료.')

# ====================================================
# 데이터셋
# ====================================================
class ConditionalLowLightDataset(Dataset):
    def __init__(self, low_dir:str, enh_dir:str, meta_file:str, transform=None, augment:bool=False):
        self.low_dir, self.enh_dir = low_dir, enh_dir
        self.low_files = sorted(os.listdir(low_dir))
        self.enh_files = sorted([f for f in os.listdir(enh_dir)
                                 if not f.startswith('mask_') and f.lower().endswith(('.jpg', '.png'))])
        with open(meta_file) as f:
            self.meta = json.load(f)
        self.transform = transform
        self.augment = augment
        self.aug = T.ColorJitter(0.3,0.3,0.3,0.1)
    def __len__(self):
        return len(self.enh_files)
    def __getitem__(self, idx:int):
        enh = self.enh_files[idx]
        low_path = os.path.join(self.low_dir, enh.split('_')[0] + '.jpg')
        enh_path = os.path.join(self.enh_dir, enh)
        mask_path = os.path.join(self.enh_dir, f"mask_{enh}")
        low, enh_img = [cv2.imread(p) for p in (low_path, enh_path)]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        low_rgb = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
        enh_rgb = cv2.cvtColor(enh_img, cv2.COLOR_BGR2RGB)
        to_t = self.transform if self.transform else (lambda x: torch.tensor(x).permute(2,0,1).float().div(255))
        low_t, enh_t = to_t(low_rgb), to_t(enh_rgb)
        if self.augment: low_t = self.aug(low_t)
        m = np.ones((IMG_H, IMG_W), np.float32) if mask is None else cv2.resize(mask,(IMG_W,IMG_H)).astype(np.float32)/255.
        m_t = torch.tensor(m).unsqueeze(0)
        md = self.meta[enh]
        cond = torch.tensor([
            md['brightness']/255.,
            *[c/255. for c in md['color_shift']],
            md['sat_shift']/255.
        ], dtype=torch.float32)
        return low_t, enh_t, cond, m_t

# ====================================================
# 학습 루프
# ====================================================

def train(low_dir:str, enh_dir:str, meta_file:str, epochs:int=1000, bs:int=8, lr:float=2e-2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tfm = T.Compose([T.ToPILImage(), T.Resize((IMG_H,IMG_W)), T.ToTensor()])
    ds = ConditionalLowLightDataset(low_dir, enh_dir, meta_file, tfm, augment=True)
    n_val = int(0.2*len(ds)); tr_ds, va_ds = random_split(ds,[len(ds)-n_val,n_val])
    tr_loader = DataLoader(tr_ds, bs, shuffle=True); va_loader = DataLoader(va_ds, bs)
    model, edge_net, sobel = UNetConditionalModel().to(device), SimpleEdgeExtractor().to(device), Sobel().to(device)
    opt = optim.Adam(model.parameters(), lr); scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt,'min',0.5,6)
    mse, perc, lpips_loss = nn.MSELoss(), VGGPerceptualLoss().to(device), lpips.LPIPS(net='vgg').to(device)
    best = float('inf'); patience=0

    def forward_pass(lo, eh, cond, msk):
        b = cond[:,0:1]; cs = cond[:,1:4]; s = cond[:,4:5]
        lo_hsv = KC.rgb_to_hsv(lo)
        lo_hsv[:,2:3,:,:] = torch.clamp(lo_hsv[:,2:3,:,:] + b.view(-1,1,1,1)*msk,0,1)
        lo_b = KC.hsv_to_rgb(lo_hsv)
        lo_bc = torch.clamp(lo_b + cs.view(-1,3,1,1)*msk, 0,1)
        gray = KC.rgb_to_grayscale(lo_bc); sobel_map = torch.norm(sobel(gray),dim=1,keepdim=True)
        learned_map = edge_net(lo_bc); struct = torch.cat([sobel_map, learned_map],1)
        residual = model(lo_bc, cs, struct)
        out_raw = torch.clamp(lo_bc + residual,0,1)
        # ▶ 사후 Saturation 후보정
        out_hsv = KC.rgb_to_hsv(out_raw)
        out_hsv[:,1:2,:,:] = torch.clamp(out_hsv[:,1:2,:,:] + s.view(-1,1,1,1)*msk, 0,1)
        out = KC.hsv_to_rgb(out_hsv)
        return out

    for epoch in range(epochs):
        model.train(); tot=0
        for lo, eh, cond, msk in tr_loader:
            lo, eh, cond, msk = [t.to(device) for t in (lo,eh,cond,msk)]
            msk = msk.unsqueeze(1)  # (B,1,H,W)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                out = forward_pass(lo, eh, cond, msk)
                loss = mse(out, eh) + perc(out, eh) + lpips_loss(out, eh).mean()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item()
        # ───── Validation
        model.eval(); vloss=0; with torch.no_grad():
            for lo, eh, cond, msk in va_loader:
                lo, eh, cond, msk = [t.to(device) for t in (lo,eh,cond,msk)]; msk=msk.unsqueeze(1)
                out = forward_pass(lo, eh, cond, msk)
                vloss += (mse(out,eh)+perc(out,eh)+lpips_loss(out,eh).mean()).item()
        vloss/=len(va_loader); print(f"Epoch {epoch+1}/{epochs} ▶ train {tot/len(tr_loader):.4f} | val {vloss:.4f}")
        sched.step(vloss); patience = patience+1 if vloss>=best else 0; best=min(best,vloss)
        if patience>15: print('Early stop'); break
    torch.save(model.state_dict(),'final.pth'); print('✅ 학습 완료')

# ====================================================
# 추론
# ====================================================

def inference(image_path:str, brightness:float, rgb_shift:list, sat_shift:float):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_bgr = cv2.imread(image_path)
    transform = T.Compose([T.ToPILImage(), T.Resize((IMG_H,IMG_W)), T.ToTensor()])
    lo = transform(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    cond = torch.tensor([[brightness/255., *[c/255. for c in rgb_shift], sat_shift/255.]],dtype=torch.float32).to(device)
    msk = torch.ones(1,1,IMG_H,IMG_W, device=device)  # 전체 영역 보정
    model = UNetConditionalModel().to(device); model.load_state_dict(torch.load('final.pth', map_location=device)); model.eval()
    edge_net, sobel = SimpleEdgeExtractor().to(device), Sobel().to(device)
    with torch.no_grad():
        b = cond[:,0:1]; cs = cond[:,1:4]; s = cond[:,4:5]
        lo_hsv = KC.rgb_to_hsv(lo); lo_hsv[:,2:3,:,:] = torch.clamp(lo_hsv[:,2:3,:,:]+b.view(-1,1,1,1),0,1)
        lo_b = KC.hsv_to_rgb(lo_hsv); lo_bc = torch.clamp(lo_b+cs.view(-1,3,1,1),0,1)
        gray = KC.rgb_to_grayscale(lo_bc); sobel_map = torch.norm(sobel(gray),1,True); struct = torch.cat([sobel_map, edge_net(lo_bc)],1)
        residual = model(lo_bc, cs, struct); out_raw = torch.clamp(lo_bc+residual,0,1)
        out_hsv = KC.rgb_to_hsv(out_raw); out_hsv[:,1:2,:,:] = torch.clamp(out_hsv[:,1:2,:,:]+s.view(-1,1,1,1),0,1)
        out = KC.hsv_to_rgb(out_hsv)[0].cpu().permute(1,2,0).numpy()*255
    cv2.imshow('Result', cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)); cv2.waitKey(0); cv2.destroyAllWindows()

# ====================================================
# CLI
# ====================================================
if __name__ == '__main__':
    mode = input('Mode(train/infer): ').strip()
    if mode=='train':
        low = input('원본 폴더: '); enh = input('보정 폴더: ')
        analyze_and_generate_metadata(low, enh)
        train(low, enh, os.path.join(enh,'metadata.json'))
    elif mode=='infer':
        p = input('이미지 경로: ')
        v = float(input('밝기 ΔV: ')); r = float(input('R shift: ')); g = float(input('G shift: ')); b = float(input('B shift: '))
        s = float(input('Sat shift: '))
        inference(p, v, [r,g,b], s)
    else:
        print('Unknown mode')
