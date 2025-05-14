
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
import kornia.color as KC

from dataset import ConditionalLowLightDataset
from model import UNetConditionalModel, SimpleEdgeExtractor, laplacian
from loss import VGGPerceptualLoss
from utils import safe_save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(low_dir, enh_dir, meta_file, epochs=1000, bs=10, lr=2e-2):
    x = [2.6667, 0.2414, 2.0, 1.5, 0.6316, 0.01108, 1.4286]
    transform = T.Compose([T.ToPILImage(), T.Resize((256,256)), T.ToTensor()])
    ds = ConditionalLowLightDataset(low_dir, enh_dir, meta_file, transform, augment=True)
    n_val = int(0.2 * len(ds))
    tr_ds, va_ds = random_split(ds, [len(ds) - n_val, n_val])
    tr = DataLoader(tr_ds, bs, shuffle=True)
    va = DataLoader(va_ds, bs)
    model = UNetConditionalModel().to(device)
    structure_model = SimpleEdgeExtractor().to(device)
    opt = optim.Adam(model.parameters(), lr)
    perc = VGGPerceptualLoss().to(device)
    mse = nn.MSELoss()
    best = float('inf')
    pat = 0

    for e in range(epochs):
        model.train()
        total_loss = 0
        for lo, eh, cond, msk in tr:
            lo, eh, cond, msk = lo.to(device), eh.to(device), cond.to(device), msk.to(device)
            b = cond[:, :1]
            cs = cond[:, 1:]

            lo_hsv = KC.rgb_to_hsv(lo)
            lo_hsv[:,2:3,:,:] = torch.clamp(lo_hsv[:,2:3,:,:] + b.view(-1,1,1,1) * msk, 0.0, 1.0)
            lo_b = KC.hsv_to_rgb(lo_hsv)
            lo_bc = torch.clamp(lo_b + cs.view(-1,3,1,1) * msk, 0.0, 1.0)

            opt.zero_grad()
            struct_map = structure_model(lo_bc)
            residual = model(lo_bc, cs, struct_map)
            out = torch.clamp(lo_bc + residual, 0.0, 1.0)

            l_mse = mse(out, eh) * x[0]
            l_per = perc(out, eh) * x[1]
            l_hf = F.l1_loss(laplacian(out), laplacian(eh)) * x[3]
            hsv_out = KC.rgb_to_hsv(out)
            hsv_gt = KC.rgb_to_hsv(eh)
            l_sat = F.l1_loss(hsv_out[:,1:2,:,:], hsv_gt[:,1:2,:,:]) * x[4]
            lab_out = KC.rgb_to_lab(out)
            lab_gt = KC.rgb_to_lab(eh)
            l_lab = F.l1_loss(lab_out[:,1:,:,:], lab_gt[:,1:,:,:]) * x[5]
            tv_h = torch.abs(out[:,:,1:,:] - out[:,:,:-1,:]).mean()
            tv_w = torch.abs(out[:,:,:,1:] - out[:,:,:,:-1]).mean()
            tv_loss = (tv_h + tv_w) * x[6]

            loss = l_mse + l_per + l_hf + l_sat + l_lab + tv_loss
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {e+1}/{epochs} Train Loss: {total_loss/len(tr):.4f}")
        if total_loss < best:
            safe_save(model, 'best.pth')
            best = total_loss
            pat = 0
        else:
            pat += 1
        if pat > 15:
            print('Early stopping triggered')
            break
