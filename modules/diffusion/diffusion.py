import torch

from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(self, x, time):
        for layer in self:
            if isinstance(layer, ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(1, 320, kernel_size=3, padding=1)),
            SwitchSequential(ResidualBlock(320, 320)),
            SwitchSequential(ResidualBlock(320, 320)),
            SwitchSequential(Downsample(320)),
            SwitchSequential(ResidualBlock(320, 640)),
            SwitchSequential(ResidualBlock(640, 640)),
            SwitchSequential(Downsample(640)),
            SwitchSequential(ResidualBlock(640, 1280)),
            SwitchSequential(ResidualBlock(1280, 1280)),
            SwitchSequential(Downsample(1280)),
            SwitchSequential(ResidualBlock(1280, 1280)),
            SwitchSequential(ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            ResidualBlock(1280, 1280),
            ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(1920, 1280), Upsample(1280)),
            SwitchSequential(ResidualBlock(1920, 640)),
            SwitchSequential(ResidualBlock(1280, 640)),
            SwitchSequential(ResidualBlock(960, 640), Upsample(640)),
            SwitchSequential(ResidualBlock(960, 320)),
            SwitchSequential(ResidualBlock(640, 320)),
            SwitchSequential(ResidualBlock(640, 320)),
        ])

    def forward(self, x, time):
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, time)
            skip_connections.append(x)

        x = self.bottleneck(x, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, time)

        return x


class FinalLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = FinalLayer(320, 1)

    def forward(self, latent, time):
        time = self.time_embedding(time)
        output = self.unet(latent, time)
        output = self.final(output)
        return output
