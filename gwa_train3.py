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
import kornia.color as KC  # for LAB conversion
import lpips  # ✅ LPIPS 추가

class SimpleEdgeExtractor(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
def make_laplacian_kernel(k):
    # k×k 라플라시안 커널: 주변에 -1, 중앙에 k*k - 1
    lap = -torch.ones((k, k), dtype=torch.float32)
    lap[k//2, k//2] = k*k - 1
    return lap

def multi_scale_hf_loss(out, gt):
    B, C, H, W = out.shape
    losses = []
    scales = [3, 5, 7]
    weights = [1.0, 0.5, 0.25]

    for k, w in zip(scales, weights):
        # 1) 라플라시안 2D 커널 생성
        lap2d = make_laplacian_kernel(k).to(out.device)            # (k,k)
        lap4d = lap2d.expand(C, 1, k, k)                           # (C,1,k,k) for group conv

        # 2) same padding
        pad = k // 2

        # 3) 그룹 컨볼루션으로 채널별 적용
        hf_out = F.conv2d(out, lap4d, padding=pad, groups=C)
        hf_gt  = F.conv2d(gt,  lap4d, padding=pad, groups=C)

        # 4) L1 손실에 스케일 가중치 곱
        losses.append(w * F.l1_loss(hf_out, hf_gt))

    return sum(losses)

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

        # ▶ RGB → HSV
        low_hsv = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        enh_hsv = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        # ▶ 동일한 mask 사용
        diff_rgb = cv2.absdiff(low_bgr, enh_bgr)
        mask     = (cv2.cvtColor(diff_rgb, cv2.COLOR_BGR2GRAY) > 15).astype(np.uint8)*255

        # ▶ V 채널(명도) 차이 계산
        V_low_px = low_hsv[...,2][mask>0]
        V_enh_px = enh_hsv[...,2][mask>0]
        if len(V_low_px) > 0:
            v_diff = float(np.mean(V_enh_px) - np.mean(V_low_px))
        else:
            v_diff = 0.0

        # 3) lo_lab를 밝기만 보정
        lo_hsv_adj = low_hsv.copy()
        lo_hsv_adj[...,2] = np.clip(lo_hsv_adj[...,2] + v_diff, 0, 255)

        # Lab → RGB 로 다시 변환
        lo_rgb_adj = cv2.cvtColor(lo_hsv_adj.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        # 4) 컬러 차이: 보정된 low와 high를 RGB 차이로 계산
        color_diff = np.mean(enh_bgr.astype(np.float32) - lo_rgb_adj, axis=(0,1)).tolist()

        metadata[enh_f] = {
            "brightness": v_diff,       # L 스케일(0–100)
            "color_shift": color_diff        # RGB 스케일 차이
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
            m = np.ones((256,256), dtype=np.float32)
        else:
            try:
                m = cv2.resize(mask, (256,256)).astype(np.float32) / 255.0
            except cv2.error:
                m = np.ones((256,256), dtype=np.float32)
        m_t = torch.tensor(m, dtype=torch.float32).unsqueeze(0).to(device)

        # Dataset.__getitem__ 내에서
        md = self.meta[enh]
        brightness = md['brightness'] / 255.0
        color_shifts = [c / 255.0 for c in md['color_shift']]
        cond = torch.tensor([brightness] + color_shifts, dtype=torch.float32).to(device)

        return low_t, enh_t, cond, m_t

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


class UNetConditionalModel(nn.Module):
    def __init__(self, cond_dim=3):
        super().__init__()
        self.cond_fc = nn.Linear(cond_dim, 256*256)
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(),
                SEBlock(out_c)
            )
        self.enc1 = block(4,64)
        self.enc2 = block(64,128)
        self.pool = nn.MaxPool2d(2)
        self.bott = block(128,256)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = block(256+128+1,128)  # 구조 map 추가
        self.dec1 = block(128+64+1,64)
        self.final = nn.Conv2d(64,3,1)


    def forward(self, x, cond, struct_map):
        b = x.size(0)
        cm = self.cond_fc(cond).view(b,1,256,256)
        x = torch.cat([x, cm], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bn = self.bott(self.pool(e2))
        
        # 🔽 구조 map을 dec에 반복해서 사용
        s_down1 = F.interpolate(struct_map, scale_factor=0.5, mode='bilinear', align_corners=False)
        s_down2 = F.interpolate(struct_map, scale_factor=1.0, mode='bilinear', align_corners=False)

        d2 = self.dec2(torch.cat([self.up(bn), e2, s_down1], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1, s_down2], dim=1))
        return self.final(d1)


# 고주파 경계 강조용 라플라시안 연산
def laplacian(x):
    kernel = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32, device=x.device)
    kernel = kernel.view(1,1,3,3).repeat(x.size(1),1,1,1)
    return F.conv2d(x, kernel, padding=1, groups=x.size(1))

# ====================================================
# 4. Perceptual Loss
# ====================================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        try:
            vgg = models.vgg16(weights=weights).features[:9].eval()
        except RuntimeError:
            cache = torch.hub.get_dir()
            ckpt = weights.url.split('/')[-1]
            path = os.path.join(cache,'checkpoints',ckpt)
            if os.path.exists(path): os.remove(path)
            vgg = models.vgg16(weights=weights).features[:9].eval()
        for p in vgg.parameters(): p.requires_grad=False
        self.vgg = vgg.to(device)
        self.crit = nn.MSELoss()
    def forward(self, x, y):
        return self.crit(self.vgg(x), self.vgg(y))

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
def train(low_dir, enh_dir, meta_file, epochs=1000, bs=10, lr=2e-2):
    transform = T.Compose([T.ToPILImage(), T.Resize((256,256)), T.ToTensor()])
    ds = ConditionalLowLightDataset(low_dir, enh_dir, meta_file, transform, augment=True)
    n_val = int(0.2 * len(ds))
    n_tr = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val])
    tr = DataLoader(tr_ds, bs, shuffle=True)
    va = DataLoader(va_ds, bs)
    model = UNetConditionalModel().to(device)
    structure_model = SimpleEdgeExtractor().to(device)
    opt = optim.Adam(model.parameters(), lr)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(tr), epochs=epochs)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=6)

    perc = VGGPerceptualLoss()
    mse = nn.MSELoss()
    lpips_loss = lpips.LPIPS(net='vgg').to(device)  # ✅ LPIPS 초기화

    best = float('inf')
    pat = 0

    for e in range(epochs):
        model.train()
        total_loss = 0
        for lo, eh, cond, msk in tr:
            b = cond[:, :1]
            cs = cond[:, 1:]
            lo, eh, cond, msk = lo.to(device), eh.to(device), cond.to(device), msk.to(device)

            lo_hsv = KC.rgb_to_hsv(lo)
            lo_hsv[:,2:3,:,:] = torch.clamp(lo_hsv[:,2:3,:,:] + b.view(-1,1,1,1) * msk, 0.0, 1.0)
            lo_b = KC.hsv_to_rgb(lo_hsv)
            lo_bc = torch.clamp(lo_b + cs.view(-1,3,1,1) * msk, 0.0, 1.0)

            opt.zero_grad()
            struct_map = structure_model(lo_bc)

            if torch.cuda.is_available():
                with torch.amp.autocast(device_type='cuda'):
                    residual = model(lo_bc, cs, struct_map)
                    out = torch.clamp(lo_bc + residual, 0.0, 1.0)
                    l_mse = mse(out, eh)
                    l_per = perc(out, eh)
                    l_lpips = lpips_loss(out, eh).mean()
                    loss = l_mse + l_per + l_lpips
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                residual = model(lo_bc, cs, struct_map)
                out = torch.clamp(lo_bc + residual, 0.0, 1.0)
                l_mse = mse(out, eh)
                l_per = perc(out, eh)
                l_lpips = lpips_loss(out, eh).mean()
                loss = l_mse + l_per + l_lpips
                loss.backward()
                opt.step()

            total_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0
        mse_loss = 0
        per_loss = 0
        lpips_eval = 0

        with torch.no_grad():
            for lo, eh, cond, msk in va:
                lo, eh, cond, msk = lo.to(device), eh.to(device), cond.to(device), msk.to(device)
                b = cond[:, :1]
                cs = cond[:, 1:]

                lo_hsv = KC.rgb_to_hsv(lo)
                lo_hsv[:,2:3,:,:] = torch.clamp(lo_hsv[:,2:3,:,:] + b.view(-1,1,1,1) * msk, 0.0, 1.0)
                lo_b = KC.hsv_to_rgb(lo_hsv)
                lo_bc = torch.clamp(lo_b + cs.view(-1,3,1,1) * msk, 0.0, 1.0)

                struct_map = structure_model(lo_bc)
                residual = model(lo_bc, cs, struct_map)
                out = torch.clamp(lo_bc + residual, 0.0, 1.0)

                l_mse = mse(out, eh)
                l_per = perc(out, eh)
                l_lpips = lpips_loss(out, eh).mean()

                val_loss += (l_mse + l_per + l_lpips).item()
                mse_loss += l_mse.item()
                per_loss += l_per.item()
                lpips_eval += l_lpips.item()

        val_loss /= len(va)
        mse_loss /= len(va)
        per_loss /= len(va)
        lpips_eval /= len(va)
        print(f"Ep {e+1}/{epochs} Train:{total_loss/len(tr):.4f} Val:{val_loss:.4f}")
        print(f"mse:{mse_loss:.4f} perc:{per_loss:.4f} lpips:{lpips_eval:.4f}")

        lr_scheduler.step(val_loss)
        if val_loss < best:
            safe_save(model, 'best.pth')
            best = val_loss
            pat = 0
        else:
            pat += 1
        if pat > 15:
            print('Early stopping triggered')
            break

    safe_save(model, 'final.pth')



# ====================================================
# 7. 추론
# ====================================================
# 전역 변수 선언

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


def inference(image_path, brightness, shifts):
    global temp_sel, mask_sel, draw_flag
    # 1. 이미지 로드 및 마우스 영역 선택
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


    model = UNetConditionalModel(cond_dim=3).to(device)
    structure_model = SimpleEdgeExtractor().to(device)
    model.load_state_dict(torch.load("final.pth", map_location=device))
    model.eval()
    structure_model.eval()  # 구조 모델도 평가 모드
    with torch.no_grad():
        struct_map = structure_model(lo_bc)
        residual   = model(lo_bc, cs, struct_map)  # 구조 map 포함
        out_tensor = torch.clamp(lo_bc + residual, 0.0, 1.0)[0]


    # 2. 입력 텐서 및 조건 벡터 생성 (정규화 포함)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256,256)),
        T.ToTensor()
    ])
    input_tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))\
                   .unsqueeze(0).to(device)
    # brightness 와 shifts 를 0~1 범위로 정규화
    cond = [brightness/255.0] + [s/255.0 for s in shifts]
    condition_tensor = torch.tensor([cond], dtype=torch.float32).to(device)

    # 3. 마스크 텐서
    mask_resized = cv2.resize(mask_sel, (input_tensor.shape[3], input_tensor.shape[2]))
    mask_tensor = (torch.from_numpy(mask_resized.astype(np.float32) / 255.0)
                   .unsqueeze(0).unsqueeze(0).to(device))  # (1,1,H,W)

    # 4. 사전 보정: 밝기 먼저, 그 다음 컬러
    b  = condition_tensor[:, :1]    # (1,1)
    cs = condition_tensor[:, 1:]    # (1,3)
    # ▶ HSV로 변환
    lo_hsv = KC.rgb_to_hsv(input_tensor)  

    # ▶ V 채널만 mask 영역에 증폭
    lo_hsv[:,2:3,:,:] = torch.clamp(
        lo_hsv[:,2:3,:,:] + b.view(-1,1,1,1) * mask_tensor,
        0.0, 1.0
    )

    # ▶ HSV → RGB
    lo_b = KC.hsv_to_rgb(lo_hsv)  

    # ▶ 그 위에 RGB 색상 이동 (잔차 학습 준비)
    lo_bc = torch.clamp(
        lo_b + cs.view(-1,3,1,1) * mask_tensor,
        0.0, 1.0
    )

    # 5. 모델 로드 및 잔차 예측
    model = UNetConditionalModel(cond_dim=3).to(device)
    model.load_state_dict(torch.load("final.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        residual   = model(lo_bc, cs)                       # 색감·디테일 잔차
        out_tensor = torch.clamp(lo_bc + residual, 0.0, 1.0)[0]  # (3,H,W)

    # 6. 텐서를 이미지로 변환 후 마스크 합성
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
        train(low, enh, os.path.join(enh, "metadata.json"))
    elif mode == "infer":
        path = input("이미지 경로: ")
        b = float(input("밝기 조정값: "))
        r = float(input("R shift: "))
        g = float(input("G shift: "))
        b2 = float(input("B shift: "))
        inference(path, b, [r, g, b2])
    else:
        print("Unknown mode")
