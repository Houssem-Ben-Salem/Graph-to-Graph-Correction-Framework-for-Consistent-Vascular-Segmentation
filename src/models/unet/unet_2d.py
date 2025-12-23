"""2D U-Net implementation for slice-based segmentation"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv2D(nn.Module):
    """Double convolution block for 2D"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down2D(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up2D(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatches
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """2D U-Net architecture for slice-by-slice processing"""
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024], 
                 bilinear=True, dropout=0.1):
        super(UNet2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder path
        self.inc = DoubleConv2D(in_channels, features[0])
        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(Down2D(features[i], features[i + 1]))
        
        # Decoder path
        self.ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(Up2D(features[i], features[i - 1], bilinear))
        
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # Encoder
        x_enc = [self.inc(x)]
        for down in self.downs:
            x_enc.append(down(x_enc[-1]))
        
        # Apply dropout to bottleneck
        x = self.dropout(x_enc[-1])
        
        # Decoder
        for i, up in enumerate(self.ups):
            x = up(x, x_enc[-(i + 2)])
        
        logits = self.outc(x)
        return logits
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)