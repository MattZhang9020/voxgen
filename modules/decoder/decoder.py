import torch

import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= torch.sqrt(torch.tensor(self.d_head, dtype=torch.float))
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residue = x
        x = self.groupnorm(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        x += residue
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class Decoder(nn.Module):
    def __init__(self, model_pram):
        super().__init__()

        self.num_parts = model_pram['num_parts']
        self.latent_dim = model_pram['latent_dim']
        self.decoder_lr = model_pram['decoder_lr']
        self.latent_lr = model_pram['latent_lr']

        self.decoder = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=1, padding=0),
            nn.Conv2d(1, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 256),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
        ])
        
        base = np.power(10, np.log10(np.prod(self.latent_dim)) / 3).astype(int)
        self.upsampler_input_dim = (1, base, base, base)
        
        self.upsampler = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            nn.Conv3d(1, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv3d(128, 1, kernel_size=3, padding=1),            
        ])

        self.latents = nn.ParameterList([nn.Parameter(torch.randn(*self.latent_dim)) for _ in range(self.num_parts)])

        self.decoder_optim = optim.AdamW(self.decoder.parameters(), lr=self.decoder_lr)
        self.latent_optim = optim.AdamW(self.latents, lr=self.latent_lr)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        for layers in self.decoder:
            x = layers(x)
        x = x.view(-1, *self.upsampler_input_dim)
        for layers in self.upsampler:
            x = layers(x)
        return x

    def train_step(self, indices, labels):
        self.decoder_optim.zero_grad(set_to_none=True)
        self.latent_optim.zero_grad(set_to_none=True)

        latents_batch = torch.stack([self.latents[i] for i in indices])
        outputs = self(latents_batch)

        loss = self.loss_fn(outputs, labels)
        loss.backward()

        self.decoder_optim.step()
        self.latent_optim.step()

        return loss.item()
