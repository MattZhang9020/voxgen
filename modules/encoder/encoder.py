import numpy as np

from ..attention import AttentionBlock
from ..residual import ResidualBlock

from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent_dim, in_ch=1, ve_ch=8, le_ch=512):
        super().__init__()

        self.latent_dim = latent_dim

        self.voxel_encoder = nn.ModuleList([
            nn.Conv3d(in_ch, ve_ch, kernel_size=2, padding=0, stride=2),
            nn.GroupNorm(2, ve_ch),
            nn.SiLU(),
            nn.Conv3d(ve_ch, ve_ch, kernel_size=2, padding=0, stride=2),
            nn.GroupNorm(2, ve_ch),
            nn.SiLU(),
            nn.Conv3d(ve_ch, ve_ch, kernel_size=2, padding=0, stride=2),
            nn.GroupNorm(2, ve_ch),
            nn.SiLU(),
            nn.Conv3d(ve_ch, le_ch, kernel_size=3, padding=1),
        ])

        base = np.ceil(np.power(10, np.log10(np.prod(self.latent_dim)) / 2)).astype(int)
        self.latent_encoder_input_dim = (le_ch, base, base)

        self.latent_encoder = nn.ModuleList([
            ResidualBlock(le_ch, le_ch),
            ResidualBlock(le_ch, le_ch),
            ResidualBlock(le_ch, le_ch),
            AttentionBlock(le_ch),
            ResidualBlock(le_ch, le_ch),
            nn.GroupNorm(32, le_ch),
            nn.SiLU(),
            nn.Conv2d(le_ch, 1, kernel_size=3, padding=1),
            nn.Conv2d(1, 1, kernel_size=1, padding=0),
        ])

    def forward(self, x):
        for layer in self.voxel_encoder:
            x = layer(x)

        x = x.view(-1, *self.latent_encoder_input_dim)

        for layer in self.latent_encoder:
            x = layer(x)
        return x
