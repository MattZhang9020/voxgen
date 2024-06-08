import torch

from torch import nn


class LatentVariables(nn.Module):
    def __init__(self, num_parts, latent_dim):
        super().__init__()

        self.num_parts = num_parts
        self.latent_dim = latent_dim

        self.latents = nn.Parameter(torch.randn(self.num_parts, 1, *self.latent_dim, requires_grad=True))

    def forward(self, indices):
        return self.latents[indices]
