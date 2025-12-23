"""Attention U-Net implementation for volumetric segmentation"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Attention Gate module for Attention U-Net"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # g: gating signal from coarser scale
        # x: skip connection from encoder
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Resize g1 to match x1 dimensions
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class DoubleConv(nn.Module):
    """Double convolution block with normalization"""
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


class UpWithAttention(nn.Module):
    """Upscaling with attention gate then double conv"""
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
            
        # Attention gate
        # F_g: channels from gating signal (upsampled features)
        # F_l: channels from skip connection (encoder features)
        if trilinear:
            self.attention = AttentionGate(F_g=in_channels, F_l=out_channels, F_int=out_channels // 2)
        else:
            self.attention = AttentionGate(F_g=in_channels // 2, F_l=out_channels, F_int=out_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Apply attention gate
        x2 = self.attention(g=x1, x=x2)
        
        # Handle size mismatches
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionUNet3D(nn.Module):
    """3D Attention U-Net architecture"""
    
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512], 
                 trilinear=True, dropout=0.1):
        super(AttentionUNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trilinear = trilinear
        
        # Encoder path
        self.inc = DoubleConv(in_channels, features[0])
        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(Down(features[i], features[i + 1]))
        
        # Decoder path with attention
        self.ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(UpWithAttention(features[i], features[i - 1], trilinear))
        
        self.outc = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        # Encoder
        x_enc = [self.inc(x)]
        for down in self.downs:
            x_enc.append(down(x_enc[-1]))
        
        # Apply dropout to bottleneck
        x = self.dropout(x_enc[-1])
        
        # Decoder with attention
        for i, up in enumerate(self.ups):
            x = up(x, x_enc[-(i + 2)])
        
        logits = self.outc(x)
        return logits
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)