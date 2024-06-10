import torch

from .utils import extract, get_time_embedding, sigmoid_beta_schedule

from torch import nn
from torch.nn import functional as F

class DiffusionTrainer(nn.Module):
    def __init__(self, model, steps, beta_start, beta_end):
        super().__init__()

        self.model = model
        self.steps = steps

        betas = sigmoid_beta_schedule(steps, beta_start, beta_end).double()
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
    
    def get_x_T(self, x_0):
        t = x_0.new_ones([x_0.shape[0], ], dtype=torch.long) * (self.steps-1)
                
        noise = torch.randn_like(x_0)
                
        x_T = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        return x_T

    def forward(self, x_0):
        t = torch.randint(self.steps, size=(x_0.shape[0], ), device=x_0.device)

        noise = torch.randn_like(x_0)

        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        loss = F.mse_loss(self.model(x_t, get_time_embedding(t)), noise)

        return loss
