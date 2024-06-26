import torch

import numpy as np

from .utils import embed_sinusoidal_position, sigmoid_beta_schedule

from torch.nn import functional as F


class DiffusionTrainer():
    def __init__(self, model, steps, beta_start, beta_end):
        self.model = model
        self.steps = steps

        betas = sigmoid_beta_schedule(steps, beta_start, beta_end).cpu().numpy()
        alphas = 1. - betas
        alphas_bar = np.cumprod(alphas)

        self.sqrt_alphas_bar = np.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = np.sqrt(1. - alphas_bar)

    def get_x_T(self, x_0):
        t = np.ones([x_0.shape[0], ], dtype=int) * (self.steps-1)
        
        sqrt_alphas_bar = torch.tensor(self.sqrt_alphas_bar[t], device=x_0.device, dtype=torch.float)
        sqrt_alphas_bar = sqrt_alphas_bar.view([t.shape[0]] + [1] * (len(x_0.shape) - 1))
        
        sqrt_one_minus_alphas_bar = torch.tensor(self.sqrt_one_minus_alphas_bar[t], device=x_0.device, dtype=torch.float)
        sqrt_one_minus_alphas_bar = sqrt_one_minus_alphas_bar.view([t.shape[0]] + [1] * (len(x_0.shape) - 1))

        noise = torch.randn_like(x_0, device=x_0.device)

        x_T = sqrt_alphas_bar * x_0 + sqrt_one_minus_alphas_bar * noise

        return x_T

    def __call__(self, x_0):
        t = np.random.randint(self.steps, size=(x_0.shape[0], ))

        noise = torch.randn_like(x_0, device=x_0.device)

        sqrt_alphas_bar = torch.tensor(self.sqrt_alphas_bar[t], device=x_0.device, dtype=torch.float)
        sqrt_alphas_bar = sqrt_alphas_bar.view([t.shape[0]] + [1] * (len(x_0.shape) - 1))
        
        sqrt_one_minus_alphas_bar = torch.tensor(self.sqrt_one_minus_alphas_bar[t], device=x_0.device, dtype=torch.float)
        sqrt_one_minus_alphas_bar = sqrt_one_minus_alphas_bar.view([t.shape[0]] + [1] * (len(x_0.shape) - 1))
        
        x_t = sqrt_alphas_bar * x_0 + sqrt_one_minus_alphas_bar * noise
        
        t_embed = embed_sinusoidal_position(t)
        t_embed = torch.tensor(t_embed, device=x_0.device, dtype=torch.float)

        loss = F.mse_loss(self.model(x_t, t_embed), noise)

        return loss
