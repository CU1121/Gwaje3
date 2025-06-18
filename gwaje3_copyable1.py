import os
import cv2
import numpy as np
import json
import torch, gc
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
import torch.nn.functional as F
import kornia.color as KC  # for LAB conversion
import lpips  # ✅ LPIPS 추가
from kornia.filters import Sobel  # Sobel 필터 추가

# ====================================================
# 글로벌 이미지 크기 설정 (H, W)
# ====================================================
IMG_H = 400  # height
IMG_W = 600  # width

class SimpleEdgeExtractor(nn.Module):
    def __init__(self, in_ch=3, out_ch=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, out_ch, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def edge_consistency_loss(pred, target, sobel):
    pred_gray = KC.rgb_to_grayscale(pred)
    target_gray = KC.rgb_to_grayscale(target)
    pred_edges = torch.norm(sobel(pred_gray), dim=1, keepdim=True)
    target_edges = torch.norm(sobel(target_gray), dim=1, keepdim=True)
    return F.l1_loss(pred_edges, target_edges)



# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def make_laplacian_kernel(k):
    lap = -torch.ones((k, k), dtype=torch.float32)
    lap[k//2, k//2] = k*k - 1
    return lap

def multi_scale_hf_loss(out, gt):
    B, C, H, W = out.shape
    losses = []
    scales = [3, 5, 7]
    weights = [1.0, 0.5, 0.25]

    for k, w in zip(scales, weights):
        lap2d = make_laplacian_kernel(k).to(out.device)
        lap4d = lap2d.expand(C, 1, k, k)
        pad = k // 2
        hf_out = F.conv2d(out, lap4d, padding=pad, groups=C)
        hf_gt  = F.conv2d(gt,  lap4d, padding=pad, groups=C)
        losses.append(w * F.l1_loss(hf_out, hf_gt))

    return sum(losses)

def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0):
    mse = F.mse_loss(x, y, reduction='none')
    mse = mse.flatten(start_dim=1).mean(dim=1)
    psnr = 10 * torch.log10(max_val**2 / (mse + 1e-8))
    return psnr.mean().item()

# ====================================================
# 1. 메타데이터 생성
# ====================================================

def analyze_and_generate_metadata(low_dir, enh_dir, save_name="metadata.json"):
    metadata = {}
    low_files = sorted(os.listdir(low_dir))
    enh_files = sorted([
        f for f in os.listdir(enh_dir)
        if not f.startswith('mask_') and f.lower().endswith(('.jpg', '.png'))
    ])

    for low_f, enh_f in zip(low_files, enh_files):
        low_bgr = cv2.imread(os.path.join(low_dir, low_f))
        enh_bgr = cv2.imread(os.path.join(enh_dir, enh_f))
        if low_bgr is None or enh_bgr is None:
            continue

        low_hsv = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        enh_hsv = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        diff_rgb = cv2.absdiff(low_bgr, enh_bgr)
        mask     = (cv2.cvtColor(diff_rgb, cv2.COLOR_BGR2GRAY) > 15).astype(np.uint8)*255

        V_low_px = low_hsv[...,2][mask>0]
        V_enh_px = enh_hsv[...,2][mask>0]
        v_diff = float(np.mean(V_enh_px) - np.mean(V_low_px)) if len(V_low_px) > 0 else 0.0

        lo_hsv_adj = low_hsv.copy()
        lo_hsv_adj[...,2] = np.clip(lo_hsv_adj[...,2] + v_diff, 0, 255)
        lo_rgb_adj = cv2.cvtColor(lo_hsv_adj.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        color_diff = np.mean(enh_bgr.astype(np.float32) - lo_rgb_adj, axis=(0,1)).tolist()

        metadata[enh_f] = {
            "brightness": v_diff,
            "color_shift": color_diff
        }
        print(v_diff, color_diff)
        cv2.imwrite(os.path.join(enh_dir, f"mask_{enh_f}"), mask)

    with open(os.path.join(enh_dir, save_name), 'w') as f:
        json.dump(metadata, f, indent=4)
    print("✅ 메타데이터 생성 완료.")


# ====================================================
# 2. 데이터셋 및 증강
# ====================================================
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
            low_t = self.transform(low_rgb).to(device)
            enh_t = self.transform(enh_rgb).to(device)
        else:
            low_t = torch.tensor(low_rgb).permute(2,0,1).float().div(255).to(device)
            enh_t = torch.tensor(enh_rgb).permute(2,0,1).float().div(255).to(device)
        if self.augment:
            low_t = self.aug(low_t)

        if mask is None:
            m = np.ones((IMG_H, IMG_W), dtype=np.float32)
        else:
            try:
                m = cv2.resize(mask, (IMG_W, IMG_H)).astype(np.float32) / 255.0
            except cv2.error:
                m = np.ones((IMG_H, IMG_W), dtype=np.float32)
        m_t = torch.tensor(m, dtype=torch.float32).unsqueeze(0).to(device)

        md = self.meta[enh]
        brightness = md['brightness'] / 255.0
        color_shifts = [c / 255.0 for c in md['color_shift']]
        cond = torch.tensor([brightness] + color_shifts, dtype=torch.float32).to(device)
        
        low_t_for_struct = low_t.unsqueeze(0) if low_t.dim() == 3 else low_t  # (1,3,H,W)
        gray = KC.rgb_to_grayscale(low_t_for_struct)
        sobel_map = torch.norm(Sobel().to(device)(gray), dim=1, keepdim=True)
        structure_model = SimpleEdgeExtractor().to(device)
        structure_model.eval()
        with torch.no_grad():
            learned_map = structure_model(low_t_for_struct)
        local_t = torch.cat([sobel_map, learned_map], dim=1).squeeze(0).to(device)  # (9,H,W)
        return low_t, enh_t, cond, m_t, local_t


# ====================================================
# 3. 모델 정의: U-Net + SE-Attention + 고주파 경계 연산
# ====================================================
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch//r, 1), nn.ReLU(),
            nn.Conv2d(ch//r, ch, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.net(x)
class GlobalInputModule(nn.Module):
    def __init__(self, cond_dim=3, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 512), nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, cond):
        return self.mlp(cond).unsqueeze(2).unsqueeze(3)

class LocalInputModule(nn.Module):
    def __init__(self, in_ch=9, out_ch=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=2, padding=1), nn.ReLU(),  # ↓ H/2
            nn.Conv2d(64, out_ch, 3, stride=2, padding=1), nn.ReLU()  # ↓ H/4
        )
    def forward(self, x):
        return self.conv(x)



class UNetConditionalModel(nn.Module):
    def __init__(self, cond_dim=4, img_h: int = IMG_H, img_w: int = IMG_W):
        super().__init__()
        img_ch = 3
        local_ch = 9

        self.img_h = img_h
        self.img_w = img_w
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(),
                SEBlock(out_c)
            )

        self.global_mlp = GlobalInputModule(cond_dim=cond_dim, out_dim=256)
        self.local_cnn = LocalInputModule(in_ch=local_ch, out_ch=256)

        self.enc1 = block(img_ch, 64)
        self.enc2 = block(64, 256)
        self.pool = nn.MaxPool2d(2)
        self.bott = block(256, 256)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = block(256 + 256, 128)
        self.dec1 = block(128 + 64, 64)
        self.final = nn.Conv2d(64, 3, 1)


    def forward(self, x, cond, local_input):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
    
        # Global MLP
        e2_pooled = self.pool(e2)
        g_feat = self.global_mlp(cond)           # (B,256,1,1)
        g_feat = g_feat.expand(-1, -1, e2_pooled.shape[2], e2_pooled.shape[3])  # ✅ (B,256,100,150)

    
        # Local CNN
        l_feat = self.local_cnn(local_input)     # (B,256,H/4,W/4)
    
        # Bottleneck
        bn_input = e2_pooled + g_feat + l_feat
        bn = self.bott(bn_input)
    
        # Decoder
        d2 = self.dec2(torch.cat([self.up(bn), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.final(d1)


def laplacian(x):
    kernel = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32, device=x.device)
    kernel = kernel.view(1,1,3,3).repeat(x.size(1),1,1,1)
    return F.conv2d(x, kernel, padding=1, groups=x.size(1))



# ====================================================
# 5. 저장 유틸
# ====================================================

def safe_save(model, path):
    tmp = path + '.tmp'
    try:
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, path)
        print(f"✅ 저장 완료: {path}")
    except Exception as e:
        if os.path.exists(tmp): os.remove(tmp)
        print(f"❌ 저장 오류: {e}")


# ====================================================
# 6. 학습 루프 (MSE + Perceptual + LPIPS Loss 추가)
# ====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import kornia.color as KC
from kornia.filters import Sobel
from tqdm import tqdm
from torchvision import models

# PSNR 계산 함수
def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0):
    mse = F.mse_loss(x, y, reduction='none')
    mse = mse.flatten(start_dim=1).mean(dim=1)
    psnr = 10 * torch.log10(max_val**2 / (mse + 1e-8))
    return psnr.mean().item()

# VGG 기반 Perceptual Loss
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights).features[:9].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.crit = nn.MSELoss()

    def forward(self, x, y):
        return self.crit(self.vgg(x), self.vgg(y))

# 혼합 손실 기반 학습 루프
def train_with_hybrid_loss(model, structure_model, train_loader, val_loader,
                           optimizer, epochs, device,
                           save_path="best.pth", save_final="final.pth"):
    mse = nn.MSELoss()
    lpips_loss = lpips.LPIPS(net='vgg').to(device)
    perc = VGGPerceptualLoss().to(device)
    sobel = Sobel().to(device)

    best_val_loss = float('inf')
    patience = 0
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(epochs):
        psnr_train_eval = 0
        model.train()
        total_loss = 0
        for lo, eh, cond, mask, local_input in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
            psnr_batch = 0
            lo, eh, cond, mask = lo.to(device), eh.to(device), cond.to(device), mask.to(device)
            b  = cond[:, :1]
            cs = cond[:, 1:]

            # Global 조정 (mask 제한 적용)
            lo_hsv = KC.rgb_to_hsv(lo)
            mask_1ch = mask[:, :1, :, :]
            lo_hsv[:, 2:3, :, :] = torch.clamp(lo_hsv[:, 2:3, :, :] + b.view(-1, 1, 1, 1) * mask_1ch, 0.0, 1.0)
            lo_b = KC.hsv_to_rgb(lo_hsv)
            lo_bc = torch.clamp(lo_b + cs.view(-1, 3, 1, 1) * mask_1ch, 0.0, 1.0)


            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                residual = model(lo_bc, cond, local_input)
                out = torch.clamp(lo_bc + residual, 0.0, 1.0)
                gray = KC.rgb_to_grayscale(out)
                sobel_map = torch.norm(sobel(gray), dim=1, keepdim=True)
                #  Sobel 기반 경계 손실 계산
                target_gray = KC.rgb_to_grayscale(eh)
                target_sobel = torch.norm(sobel(target_gray), dim=1, keepdim=True)
                edge_loss = F.l1_loss(sobel_map, target_sobel)
            
                # Global 손실 구성
                mse_g = mse(out, eh)
                perc_g = perc(out, eh)
                lpips_g = lpips_loss(out, eh).mean()
                edge_g = edge_consistency_loss(out, eh, sobel)
                loss_global = 30 * mse_g + 1.5 * perc_g + 1.5 * lpips_g + 30 * edge_g
                
                # Local 손실 구성
                mask_rgb = mask.expand_as(out)  # (B,1,H,W) → (B,3,H,W)
                out_mask = out * mask_rgb
                eh_mask = eh * mask_rgb
                mse_l = mse(out_mask, eh_mask)
                perc_l = perc(out_mask, eh_mask)
                lpips_l = lpips_loss(out_mask, eh_mask).mean()
                edge_l = edge_consistency_loss(out_mask, eh_mask, sobel)
                loss_local = 30 * mse_l + 1.5 * perc_l + 1.5 * lpips_l + 30 * edge_l
                
                # 조건 분기
                has_global = True
                has_local = (mask.sum() > 0).item()
                
                if has_global and has_local:
                    loss = loss_global + loss_local
                elif has_global:
                    loss = loss_global
                elif has_local:
                    loss = loss_local
                else:
                    loss = mse(out, eh)  # fallback


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            psnr_batch = psnr(out, eh)
            psnr_train_eval += psnr_batch


        # 검증
        model.eval()
        mse_loss, perc_loss, lp_loss, ed_loss = 0, 0, 0, 0
        val_loss, psnr_eval = 0, 0
        global_loss, local_loss = 0, 0
        with torch.no_grad():
            for lo, eh, cond, mask, local_input  in val_loader:
                lo, eh, cond, mask = lo.to(device), eh.to(device), cond.to(device), mask.to(device)
                b = cond[:, :1]
                cs = cond[:, 1:]

                lo_hsv = KC.rgb_to_hsv(lo)
                mask_1ch = mask[:, :1, :, :]
                lo_hsv[:, 2:3, :, :] = torch.clamp(lo_hsv[:, 2:3, :, :] + b.view(-1, 1, 1, 1) * mask_1ch, 0.0, 1.0)
                lo_b = KC.hsv_to_rgb(lo_hsv)
                lo_bc = torch.clamp(lo_b + cs.view(-1, 3, 1, 1) * mask_1ch, 0.0, 1.0)

                residual = model(lo_bc, cond, local_input)

                residual = model(lo_bc, cond, local_input)

                out = torch.clamp(lo_bc + residual, 0.0, 1.0)
                gray = KC.rgb_to_grayscale(out)
                sobel_map = torch.norm(sobel(gray), dim=1, keepdim=True)
                #  Sobel 기반 경계 손실 계산
                target_gray = KC.rgb_to_grayscale(eh)
                target_sobel = torch.norm(sobel(target_gray), dim=1, keepdim=True)
                edge_loss = F.l1_loss(sobel_map, target_sobel)

                m = mse(out,eh)
                p = perc(out,eh)
                l = lpips_loss(out,eh).mean()
                e = F.l1_loss(sobel_map, target_sobel)

                loss_global = 30 * m + 1.5 * p + 1.5 * l + 30 * e
                mse_loss += m
                perc_loss += p
                lp_loss += l
                ed_loss += e
                
                loss_local = 90 * ((out - eh) ** 2 * mask).mean()
                loss = loss_global + loss_local
                global_loss += loss_global
                local_loss += loss_local

                val_loss += loss.item()
                psnr_eval += psnr(out, eh)

        val_loss /= len(val_loader)
        psnr_eval /= len(val_loader)
        mse_loss /= len(val_loader)
        perc_loss /= len(val_loader)
        lp_loss /= len(val_loader)
        ed_loss /= len(val_loader)
        global_loss /= len(val_loader)
        local_loss /= len(val_loader)
        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f} | Val: {val_loss:.4f} | PSNR: {psnr_eval:.2f}dB")
        print(f" mse : {mse_loss}, perc : {perc_loss}, lp : {lp_loss}, global : {global_loss}, local : {local_loss}, edge : {ed_loss}")
        print(f" Val PSNR: {psnr_eval:.2f}dB")

        lr_scheduler.step(val_loss)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), save_path)
            best_val_loss = val_loss
            patience = 0
            print(f"✅ Saved best model to {save_path}")
        else:
            patience += 1
            if patience > 15:
                print("🛑 Early stopping triggered.")
                break

    torch.save(model.state_dict(), save_final)
    print(f" Final model saved to {save_final}")

# ====================================================
# 7. 추론
# ====================================================

draw_flag = False
mask_sel = None
temp_sel = None


def draw_sel(event, x, y, flags, param):
    global draw_flag, mask_sel, temp_sel
    if event == cv2.EVENT_LBUTTONDOWN:
        draw_flag = True
    elif event == cv2.EVENT_MOUSEMOVE and draw_flag:
        cv2.circle(temp_sel, (x, y), 20, (0, 255, 0), -1)
        cv2.circle(mask_sel, (x, y), 20, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        draw_flag = False


def inference(image_path, brightness, shifts, local_brightness=0.0):
    global temp_sel, mask_sel, draw_flag

    image = cv2.imread(image_path)
    temp_sel = image.copy()
    mask_sel = np.zeros(image.shape[:2], dtype=np.uint8)
    draw_flag = False

    cv2.namedWindow("영역 선택 (q: 완료)")
    cv2.setMouseCallback("영역 선택 (q: 완료)", draw_sel)
    while True:
        cv2.imshow("영역 선택 (q: 완료)", temp_sel)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    local_brightness = float(input("Local 밝기 조정값을 입력하세요 (0~255): "))

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor()
    ])
    input_tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    cond = [brightness/255.0] + [s/255.0 for s in shifts]
    condition_tensor = torch.tensor([cond], dtype=torch.float32).to(device)

    mask_resized = cv2.resize(mask_sel, (input_tensor.shape[3], input_tensor.shape[2]))
    mask_tensor = (torch.from_numpy(mask_resized.astype(np.float32) / 255.0)
                   .unsqueeze(0).unsqueeze(0).to(device))

    b  = condition_tensor[:, :1]
    cs = condition_tensor[:, 1:]

    lo_hsv = KC.rgb_to_hsv(input_tensor)
    lo_hsv[:,2:3,:,:] = torch.clamp(lo_hsv[:,2:3,:,:] + b.view(-1,1,1,1) * mask_tensor, 0.0, 1.0)
    lo_b = KC.hsv_to_rgb(lo_hsv)
    lo_bc = torch.clamp(lo_b + cs.view(-1,3,1,1) * mask_tensor, 0.0, 1.0)
    if local_brightness != 0.0:
        lo_hsv_local = KC.rgb_to_hsv(lo_bc.clone())
        lo_hsv_local[:,2:3,:,:] = torch.clamp(lo_hsv_local[:,2:3,:,:] + (local_brightness / 255.0) * mask_tensor, 0.0, 1.0)
        lo_bc = KC.hsv_to_rgb(lo_hsv_local)

    model = UNetConditionalModel(cond_dim=4, img_h=IMG_H, img_w=IMG_W).to(device)
    structure_model = SimpleEdgeExtractor().to(device)
    sobel = Sobel().to(device)

    model.load_state_dict(torch.load("final.pth", map_location=device))
    model.eval()
    structure_model.eval()
    sobel.eval()

    with torch.no_grad():
        gray = KC.rgb_to_grayscale(lo_bc)
        sobel_map = torch.norm(sobel(gray), dim=1, keepdim=True)
        learned_map = structure_model(lo_bc)
        local_input_tensor = torch.cat([sobel_map, learned_map], dim=1)  # (1,9,H,W)
        
        residual = model(lo_bc, condition_tensor, local_input_tensor)
        out_tensor = torch.clamp(lo_bc + residual, 0.0, 1.0)[0]

    output_img = (out_tensor.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    mask_full  = cv2.resize(mask_sel, (image.shape[1], image.shape[0]))
    mask_3ch   = np.stack([mask_full]*3, axis=2)
    result     = np.where(mask_3ch==255, output_bgr, image)

    cv2.imshow("AI 보정 결과", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mode = input("Mode(train/infer): ")
    if mode == "train":
        low = input("원본 폴더: ")
        enh = input("보정 폴더: ")
        analyze_and_generate_metadata(low, enh)
        transform = T.Compose([T.ToPILImage(), T.Resize((IMG_H, IMG_W)), T.ToTensor()])
        ds = ConditionalLowLightDataset(low, enh, os.path.join(enh, "metadata.json"), transform, augment=True)
        n_val = int(0.2 * len(ds))
        n_tr = len(ds) - n_val
        tr_ds, va_ds = random_split(ds, [n_tr, n_val])
        train_loader = DataLoader(tr_ds, batch_size=10, shuffle=True)
        val_loader = DataLoader(va_ds, batch_size=10)
        
        model = UNetConditionalModel(cond_dim=4).to(device)
        structure_model = SimpleEdgeExtractor().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        train_with_hybrid_loss(model, structure_model, train_loader, val_loader, optimizer, epochs=1000, device=device)

    elif mode == "infer":
        path = input("이미지 경로: ")
        b = float(input("밝기 조정값: "))
        r = float(input("R shift: "))
        g = float(input("G shift: "))
        b2 = float(input("B shift: "))
        inference(path, b, [r, g, b2])
    else:
        print("Unknown mode")
