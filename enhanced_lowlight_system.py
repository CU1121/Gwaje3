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
import kornia.color as KC
import lpips
from kornia.filters import Sobel
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import warnings
warnings.filterwarnings('ignore')

# Global image size settings
IMG_H = 400
IMG_W = 600

class ImprovedEdgeExtractor(nn.Module):
    """Enhanced edge extraction with residual connections"""
    def __init__(self, in_ch=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        out = self.sigmoid(self.conv4(x3))
        return out

class ImprovedSEBlock(nn.Module):
    """Enhanced SE Block with more sophisticated attention"""
    def __init__(self, ch, r=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch//r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch//r, ch, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_pool = self.fc(self.global_pool(x))
        max_pool = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_pool + max_pool)
        return x * attention

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.se = ImprovedSEBlock(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.se(out)
        return self.relu(out + residual)

class ImprovedUNetConditionalModel(nn.Module):
    """Enhanced U-Net with better architecture"""
    def __init__(self, cond_dim=4, img_h=IMG_H, img_w=IMG_W):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        
        # Condition embedding with learnable projection
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, img_h * img_w),
            nn.Tanh()
        )
        
        # Encoder
        self.enc1 = self._make_encoder_block(4, 64)
        self.enc2 = self._make_encoder_block(64, 128)  
        self.enc3 = self._make_encoder_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256)
        )
        
        # Decoder with skip connections
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self._make_decoder_block(256 + 128 + 2, 128)
        self.dec2 = self._make_decoder_block(128 + 64 + 2, 64)
        self.dec1 = self._make_decoder_block(64 + 64 + 2, 64)
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1),
            nn.Tanh()  # Output in [-1, 1] range for residual
        )
        
    def _make_encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ImprovedSEBlock(out_ch)
        )
    
    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ImprovedSEBlock(out_ch)
        )
    
    def forward(self, x, cond, struct_map):
        b = x.size(0)
        # Enhanced condition embedding
        cond_map = self.cond_embed(cond).view(b, 1, self.img_h, self.img_w)
        
        # Encoder path
        x_in = torch.cat([x, cond_map], dim=1)
        e1 = self.enc1(x_in)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        bn = self.bottleneck(self.pool(e3))
        
        # Multi-scale structure maps
        s_1 = F.interpolate(struct_map, scale_factor=0.25, mode='bilinear', align_corners=False)
        s_2 = F.interpolate(struct_map, scale_factor=0.5, mode='bilinear', align_corners=False)
        s_3 = struct_map
        
        # Decoder path with skip connections
        d3 = self.dec3(torch.cat([self.up(bn), e3, s_1], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2, s_2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1, s_3], dim=1))
        
        return self.final(d1)

class ColorLoss(nn.Module):
    """Color consistency loss in LAB space"""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        pred_lab = KC.rgb_to_lab(pred)
        target_lab = KC.rgb_to_lab(target)
        
        # L channel loss (brightness)
        l_loss = F.l1_loss(pred_lab[:, 0:1], target_lab[:, 0:1])
        # AB channels loss (color)
        ab_loss = F.l1_loss(pred_lab[:, 1:3], target_lab[:, 1:3])
        
        return l_loss + ab_loss * 0.5

def enhanced_multi_scale_loss(pred, target, scales=[3, 5, 7], weights=[1.0, 0.5, 0.25]):
    """Enhanced multi-scale loss with gradient information"""
    losses = []
    
    for scale, weight in zip(scales, weights):
        # Laplacian kernel
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                            dtype=torch.float32, device=pred.device)
        kernel = kernel.view(1, 1, 3, 3).repeat(pred.size(1), 1, 1, 1)
        
        # Apply convolution
        pred_hf = F.conv2d(pred, kernel, padding=1, groups=pred.size(1))
        target_hf = F.conv2d(target, kernel, padding=1, groups=pred.size(1))
        
        # Scale-specific loss
        hf_loss = F.l1_loss(pred_hf, target_hf)
        losses.append(weight * hf_loss)
    
    return sum(losses)

class EnhancedLossFunction(nn.Module):
    """Combined loss function with multiple components"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.color_loss = ColorLoss()
        
        # Perceptual loss
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        
        # LPIPS loss
        self.lpips = lpips.LPIPS(net='vgg')
        
    def forward(self, pred, target, mask=None):
        # Basic reconstruction losses
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        
        # Perceptual loss
        pred_feat = self.vgg(pred)
        target_feat = self.vgg(target)
        perceptual_loss = self.mse(pred_feat, target_feat)
        
        # LPIPS loss
        lpips_loss = self.lpips(pred, target).mean()
        
        # Color consistency loss
        color_loss = self.color_loss(pred, target)
        
        # High-frequency loss
        hf_loss = enhanced_multi_scale_loss(pred, target)
        
        # Weighted combination
        total_loss = (mse_loss * 10 + 
                     l1_loss * 5 +
                     perceptual_loss * 2 +
                     lpips_loss * 1 +
                     color_loss * 3 +
                     hf_loss * 2)
        
        return total_loss, {
            'mse': mse_loss.item(),
            'l1': l1_loss.item(), 
            'perceptual': perceptual_loss.item(),
            'lpips': lpips_loss.item(),
            'color': color_loss.item(),
            'hf': hf_loss.item()
        }

def enhanced_psnr(pred, target, max_val=1.0):
    """Enhanced PSNR calculation"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))

def enhanced_ssim(pred, target, window_size=11):
    """Simplified SSIM calculation"""
    mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

class EnhancedConditionalLowLightDataset(Dataset):
    """Enhanced dataset with better augmentation"""
    def __init__(self, low_dir, enh_dir, meta_file, transform=None, augment=False):
        self.low_dir = low_dir
        self.enh_dir = enh_dir
        self.meta_file = meta_file
        self.transform = transform
        self.augment = augment
        
        # Load file lists
        self.low_files = sorted([f for f in os.listdir(low_dir) 
                               if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.enh_files = sorted([f for f in os.listdir(enh_dir) 
                               if not f.startswith('mask_') and 
                               f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        # Load metadata
        with open(meta_file) as f:
            self.meta = json.load(f)
            
        # Enhanced augmentation
        if augment:
            self.augment_transform = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=5),
            ])
    
    def __len__(self):
        return len(self.enh_files)
    
    def __getitem__(self, idx):
        enh_file = self.enh_files[idx]
        
        # Find corresponding low-light image
        base_name = enh_file.split('_')[0] + '.jpg'
        if base_name not in self.low_files:
            base_name = enh_file  # fallback
            
        low_path = os.path.join(self.low_dir, base_name)
        enh_path = os.path.join(self.enh_dir, enh_file)
        mask_path = os.path.join(self.enh_dir, f"mask_{enh_file}")
        
        # Load images
        try:
            low_img = cv2.imread(low_path)
            enh_img = cv2.imread(enh_path)
            
            if low_img is None or enh_img is None:
                raise FileNotFoundError(f"Cannot load images: {low_path} or {enh_path}")
                
            # Convert to RGB
            low_rgb = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
            enh_rgb = cv2.cvtColor(enh_img, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            if self.transform:
                low_tensor = self.transform(low_rgb)
                enh_tensor = self.transform(enh_rgb)
            else:
                low_tensor = torch.from_numpy(low_rgb).permute(2,0,1).float() / 255.0
                enh_tensor = torch.from_numpy(enh_rgb).permute(2,0,1).float() / 255.0
                
            # Apply augmentation
            if self.augment and hasattr(self, 'augment_transform'):
                # Convert to PIL for augmentation
                low_pil = T.ToPILImage()(low_tensor)
                low_tensor = T.ToTensor()(self.augment_transform(low_pil))
                
            # Load mask
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (IMG_W, IMG_H))
                mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                mask_tensor = torch.ones(1, IMG_H, IMG_W)
                
            # Load metadata
            if enh_file in self.meta:
                metadata = self.meta[enh_file]
                brightness = metadata['brightness'] / 255.0
                color_shifts = [c / 255.0 for c in metadata['color_shift']]
            else:
                brightness = 0.0
                color_shifts = [0.0, 0.0, 0.0]
                
            # Combine conditions
            condition = torch.tensor([brightness] + color_shifts, dtype=torch.float32)
            
            return low_tensor, enh_tensor, condition, mask_tensor
            
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            # Return dummy data
            dummy_tensor = torch.zeros(3, IMG_H, IMG_W)
            dummy_condition = torch.zeros(4)
            dummy_mask = torch.ones(1, IMG_H, IMG_W)
            return dummy_tensor, dummy_tensor, dummy_condition, dummy_mask

def enhanced_train(low_dir, enh_dir, meta_file, epochs=100, batch_size=8, lr=1e-3):
    """Enhanced training function with better monitoring"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Data preparation
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor()
    ])
    
    dataset = EnhancedConditionalLowLightDataset(
        low_dir, enh_dir, meta_file, transform, augment=True
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)
    
    # Model setup
    model = ImprovedUNetConditionalModel(cond_dim=4).to(device)
    edge_model = ImprovedEdgeExtractor().to(device)
    sobel = Sobel().to(device)
    
    # Loss and optimizer
    criterion = EnhancedLossFunction().to(device)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(edge_model.parameters()), 
        lr=lr, weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 0
    max_patience = 15
    
    print("Starting enhanced training...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        edge_model.train()
        total_train_loss = 0
        train_metrics = {'mse': 0, 'l1': 0, 'perceptual': 0, 'lpips': 0, 'color': 0, 'hf': 0}
        
        for batch_idx, (low, enh, cond, mask) in enumerate(train_loader):
            low, enh, cond, mask = low.to(device), enh.to(device), cond.to(device), mask.to(device)
            
            # Preprocessing with enhanced color correction
            brightness = cond[:, :1]
            color_shifts = cond[:, 1:]
            
            # Apply brightness and color adjustments
            low_hsv = KC.rgb_to_hsv(low)
            low_hsv[:, 2:3, :, :] = torch.clamp(
                low_hsv[:, 2:3, :, :] + brightness.view(-1, 1, 1, 1) * mask * 2.0,
                0.0, 1.0
            )
            low_adjusted = KC.hsv_to_rgb(low_hsv)
            low_adjusted = torch.clamp(
                low_adjusted + color_shifts.view(-1, 3, 1, 1) * mask,
                0.0, 1.0
            )
            
            # Structure extraction
            gray = KC.rgb_to_grayscale(low_adjusted)
            sobel_edges = torch.norm(sobel(gray), dim=1, keepdim=True)
            learned_edges = edge_model(low_adjusted)
            structure_map = torch.cat([sobel_edges, learned_edges], dim=1)
            
            # Forward pass
            optimizer.zero_grad()
            residual = model(low_adjusted, color_shifts, structure_map)
            output = torch.clamp(low_adjusted + residual * 0.5, 0.0, 1.0)
            
            # Loss calculation
            loss, metrics = criterion(output, enh, mask)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            for key in train_metrics:
                train_metrics[key] += metrics[key]
                
        # Validation phase
        model.eval()
        edge_model.eval()
        total_val_loss = 0
        val_psnr = 0
        val_ssim = 0
        
        with torch.no_grad():
            for low, enh, cond, mask in val_loader:
                low, enh, cond, mask = low.to(device), enh.to(device), cond.to(device), mask.to(device)
                
                brightness = cond[:, :1]
                color_shifts = cond[:, 1:]
                
                low_hsv = KC.rgb_to_hsv(low)
                low_hsv[:, 2:3, :, :] = torch.clamp(
                    low_hsv[:, 2:3, :, :] + brightness.view(-1, 1, 1, 1) * mask * 2.0,
                    0.0, 1.0
                )
                low_adjusted = KC.hsv_to_rgb(low_hsv)
                low_adjusted = torch.clamp(
                    low_adjusted + color_shifts.view(-1, 3, 1, 1) * mask,
                    0.0, 1.0
                )
                
                gray = KC.rgb_to_grayscale(low_adjusted)
                sobel_edges = torch.norm(sobel(gray), dim=1, keepdim=True)
                learned_edges = edge_model(low_adjusted)
                structure_map = torch.cat([sobel_edges, learned_edges], dim=1)
                
                residual = model(low_adjusted, color_shifts, structure_map)
                output = torch.clamp(low_adjusted + residual * 0.5, 0.0, 1.0)
                
                val_loss, _ = criterion(output, enh, mask)
                total_val_loss += val_loss.item()
                
                val_psnr += enhanced_psnr(output, enh)
                val_ssim += enhanced_ssim(output, enh)
        
        # Update learning rate
        scheduler.step()
        
        # Calculate averages
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_psnr = val_psnr / len(val_loader)
        avg_ssim = val_ssim / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Val PSNR: {avg_psnr:.2f}dB, Val SSIM: {avg_ssim:.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'edge_model_state_dict': edge_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss
            }, 'best_enhanced_model.pth')
            print("✅ Best model saved!")
        else:
            patience += 1
            
        if patience >= max_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'edge_model_state_dict': edge_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, 'final_enhanced_model.pth')
    
    print("✅ Training completed!")
    return train_losses, val_losses

# Example usage
if __name__ == "__main__":
    # Training example
    train_losses, val_losses = enhanced_train(
        low_dir="path/to/low_images",
        enh_dir="path/to/enhanced_images", 
        meta_file="path/to/metadata.json",
        epochs=100,
        batch_size=8,
        lr=1e-3
    )