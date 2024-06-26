import torch

from ..attention import SelfAttention

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
    def __init__(self, in_channels, out_channels, n_time=640):
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


class AttentionBlock(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))   # (n, c, hw)
        x = x.transpose(-1, -2).contiguous()     # (n, hw, c)

        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        residue_short = x
        x = self.layernorm_2(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2).contiguous()   # (n, c, hw)
        x = x.view((n, c, h, w))  # (n, c, h, w)

        return self.conv_output(x) + residue_long


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
            elif isinstance(layer, AttentionBlock):
                x = layer(x)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(1, 160, kernel_size=3, padding=1)),
            SwitchSequential(ResidualBlock(160, 160), AttentionBlock(8, 20)),
            SwitchSequential(ResidualBlock(160, 160), AttentionBlock(8, 20)),
            SwitchSequential(Downsample(160)),
            SwitchSequential(ResidualBlock(160, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            SwitchSequential(Downsample(320)),
            SwitchSequential(ResidualBlock(320, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(640, 640), AttentionBlock(8, 80)),
            SwitchSequential(Downsample(640)),
            SwitchSequential(ResidualBlock(640, 640)),
            SwitchSequential(ResidualBlock(640, 640)),
        ])
        
        self.bottleneck = SwitchSequential(
            ResidualBlock(640, 640),
            AttentionBlock(8, 80),
            ResidualBlock(640, 640),
        )
        
        self.decoders = nn.ModuleList([
            SwitchSequential(ResidualBlock(1280, 640)),
            SwitchSequential(ResidualBlock(1280, 640)),
            SwitchSequential(ResidualBlock(1280, 640), Upsample(640)),
            SwitchSequential(ResidualBlock(1280, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(1280, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(960, 640), AttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(ResidualBlock(960, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(480, 320), AttentionBlock(8, 40), Upsample(320)),
            SwitchSequential(ResidualBlock(480, 160), AttentionBlock(8, 20)),
            SwitchSequential(ResidualBlock(320, 160), AttentionBlock(8, 20)),
            SwitchSequential(ResidualBlock(320, 160), AttentionBlock(8, 20)),
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
        self.time_embedding = TimeEmbedding(160)
        self.unet = UNet()
        self.final = FinalLayer(160, 1)

    def forward(self, latent, time):
        time = self.time_embedding(time)
        output = self.unet(latent, time)
        output = self.final(output)
        return output
