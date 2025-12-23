"""3D U-Net implementation for volumetric segmentation"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            # After concatenation: in_channels (upsampled) + out_channels (skip) = total input to conv
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # After concatenation: in_channels//2 (upsampled) + out_channels (skip) = total input to conv
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatches
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net architecture for volumetric medical image segmentation"""
    
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512], 
                 trilinear=True, dropout=0.1):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trilinear = trilinear
        
        # Encoder path
        self.inc = DoubleConv(in_channels, features[0])
        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(Down(features[i], features[i + 1]))
        
        # Decoder path
        self.ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(Up(features[i], features[i - 1], trilinear))
        
        self.outc = OutConv(features[0], out_channels)
        self.dropout = nn.Dropout3d(dropout)

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