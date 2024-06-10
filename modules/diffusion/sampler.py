import torch

import numpy as np

from .utils import extract, get_time_embedding, sigmoid_beta_schedule

from torch import nn
from torch.nn import functional as F


class VanillaDiffusionSampler(nn.Module):
    def __init__(self, model, steps, beta_start, beta_end):
        super().__init__()

        self.model = model

        self.register_buffer('betas', sigmoid_beta_schedule(steps, beta_start, beta_end).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:steps]

        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer('sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))
        
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer('posterior_log_var_clipped', torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        
        self.register_buffer('posterior_mean_coef1', torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer('posterior_mean_coef2', torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        model_log_var = torch.log(torch.cat([self.posterior_var[1:2], self.betas[1:]]))
        model_log_var = extract(model_log_var, t, x_t.shape)

        eps = self.model(x_t, get_time_embedding(t))
        
        x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
        
        model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        
        return model_mean, model_log_var

    def forward(self, x_t, time_step):
        t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
                        
        mean, log_var = self.p_mean_variance(x_t, t)
        
        if time_step > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = 0
            
        x_t = mean + torch.exp(0.5 * log_var) * noise
        
        return torch.clip(x_t, -1, 1)


class KEulerDiffusionSampler(nn.Module):
    def __init__(self, model, beta_start, beta_end, inference_steps=50, training_steps=1000):
        super().__init__()
        
        self.model = model

        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, training_steps, dtype=np.float64) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        timesteps = np.linspace(training_steps - 1, 0, inference_steps)
        log_sigmas = np.log(sigmas)
        log_sigmas = np.interp(timesteps, range(training_steps), log_sigmas)
        sigmas = np.exp(log_sigmas)
        sigmas = np.append(sigmas, 0)
        
        self.register_buffer('sigmas', torch.tensor(sigmas))
        
        self.initial_scale = sigmas.max()
    
    def _get_input_scale(self, time_step):
        sigma = self.sigmas[time_step]
        return 1 / (sigma ** 2 + 1) ** 0.5
    
    def forward(self, x_T, time_step):
        if time_step == 0:
            x_T = x_T * self.initial_scale
            
        t = x_T.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
        
        x_t = x_T * self._get_input_scale(t)
        eps = self.model(x_t, get_time_embedding(t))
                
        sigma_from = self.sigmas[t]
        sigma_to = self.sigmas[t + 1]
        
        x_T = x_T + eps * (sigma_to - sigma_from)
        
        return x_T