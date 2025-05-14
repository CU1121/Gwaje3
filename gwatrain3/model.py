
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.dec2 = block(256+128+1,128)
        self.dec1 = block(128+64+1,64)
        self.final = nn.Conv2d(64,3,1)

    def forward(self, x, cond, struct_map):
        b = x.size(0)
        cm = self.cond_fc(cond).view(b,1,256,256)
        x = torch.cat([x, cm], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bn = self.bott(self.pool(e2))
        s_down1 = F.interpolate(struct_map, scale_factor=0.5, mode='bilinear', align_corners=False)
        s_down2 = F.interpolate(struct_map, scale_factor=1.0, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([self.up(bn), e2, s_down1], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1, s_down2], dim=1))
        return self.final(d1)

class SimpleEdgeExtractor(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def laplacian(x):
    kernel = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32, device=x.device)
    kernel = kernel.view(1,1,3,3).repeat(x.size(1),1,1,1)
    return F.conv2d(x, kernel, padding=1, groups=x.size(1))
