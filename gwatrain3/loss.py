
import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        try:
            vgg = models.vgg16(weights=weights).features[:9].eval()
        except RuntimeError:
            import os
            import torch.hub
            cache = torch.hub.get_dir()
            ckpt = weights.url.split('/')[-1]
            path = os.path.join(cache,'checkpoints',ckpt)
            if os.path.exists(path): os.remove(path)
            vgg = models.vgg16(weights=weights).features[:9].eval()
        for p in vgg.parameters(): p.requires_grad=False
        self.vgg = vgg
        self.crit = nn.MSELoss()

    def forward(self, x, y):
        return self.crit(self.vgg(x), self.vgg(y))
