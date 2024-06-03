import numpy as np

from ..attention import AttentionBlock
from ..residual import ResidualBlock

from torch import nn


class Decoder(nn.Module):
    def __init__(self, latent_dim, in_ch=1, ld_ch=512, vd_ch=128):
        super().__init__()

        self.latent_dim = latent_dim

        self.latent_decoder = nn.ModuleList([
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
            nn.Conv2d(in_ch, ld_ch, kernel_size=3, padding=1),
            ResidualBlock(ld_ch, ld_ch),
            AttentionBlock(ld_ch),
            ResidualBlock(ld_ch, ld_ch),
            ResidualBlock(ld_ch, ld_ch),
            ResidualBlock(ld_ch, ld_ch),
            nn.GroupNorm(32, ld_ch),
            nn.SiLU(),
            nn.Conv2d(ld_ch, vd_ch, kernel_size=3, padding=1),
        ])

        base = np.ceil(np.power(10, np.log10(np.prod(self.latent_dim)) / 3)).astype(int)
        self.voxel_decoder_input_dim = (vd_ch, base, base, base)

        self.voxel_decoder = nn.ModuleList([
            nn.ConvTranspose3d(vd_ch, vd_ch, kernel_size=2, padding=0, stride=2),
            nn.GroupNorm(2, vd_ch),
            nn.SiLU(),
            nn.ConvTranspose3d(vd_ch, vd_ch, kernel_size=2, padding=0, stride=2),
            nn.GroupNorm(2, vd_ch),
            nn.SiLU(),
            nn.ConvTranspose3d(vd_ch, vd_ch, kernel_size=2, padding=0, stride=2),
            nn.GroupNorm(2, vd_ch),
            nn.SiLU(),
            nn.Conv3d(vd_ch, 1, kernel_size=1, padding=0),
        ])

    def forward(self, x):
        for layers in self.latent_decoder:
            x = layers(x)

        x = x.view(-1, *self.voxel_decoder_input_dim)

        for layers in self.voxel_decoder:
            x = layers(x)
        return x
