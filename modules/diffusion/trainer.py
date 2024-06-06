import torch

from torch import nn
from torch.nn import functional as F

from .utils import extract, get_time_embedding


class DiffusionTrainer(nn.Module):
    def __init__(self, model, beta_start, beta_end, steps):
        super().__init__()

        self.model = model
        self.steps = steps

        self.register_buffer('betas', torch.linspace(beta_start, beta_end, steps).double())

        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        t = torch.randint(self.steps, size=(x_0.shape[0], ), device=x_0.device)

        noise = torch.randn_like(x_0)

        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        loss = F.mse_loss(self.model(x_t, get_time_embedding(t)), noise)

        return loss