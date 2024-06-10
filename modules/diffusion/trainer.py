import torch

import numpy as np

from .utils import get_time_embedding, sigmoid_beta_schedule

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
        t = x_0.new_ones([x_0.shape[0], ], dtype=torch.long) * (self.steps-1)

        noise = torch.randn_like(x_0, device=x_0.device)

        x_T = self.sqrt_alphas_bar[t] * x_0 + self.sqrt_one_minus_alphas_bar[t] * noise

        return x_T

    def __call__(self, x_0):
        t = torch.randint(self.steps, size=(x_0.shape[0], ), device=x_0.device)

        noise = torch.randn_like(x_0, device=x_0.device)

        x_t = self.sqrt_alphas_bar[t] * x_0 + self.sqrt_one_minus_alphas_bar[t] * noise

        loss = F.mse_loss(self.model(x_t, get_time_embedding(t)), noise)

        return loss
