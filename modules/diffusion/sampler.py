import torch

import numpy as np

from .utils import embed_sinusoidal_position, sigmoid_beta_schedule


class KEulerDiffusionSampler():
    def __init__(self, model, beta_start, beta_end, inference_steps=50, training_steps=1000):        
        self.model = model
        
        self.timesteps = np.linspace(training_steps - 1, 0, inference_steps)

        betas = sigmoid_beta_schedule(training_steps, beta_start, beta_end).cpu().numpy()
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        log_sigmas = np.log(sigmas)
        log_sigmas = np.interp(self.timesteps, range(training_steps), log_sigmas)
        sigmas = np.exp(log_sigmas)
        sigmas = np.append(sigmas, 0)
        
        self.sigmas = sigmas
        
        self.initial_scale = sigmas.max()
    
    def _get_input_scale(self, time_step):
        sigma = self.sigmas[time_step]
        return 1 / (sigma ** 2 + 1) ** 0.5
    
    def __call__(self, x_T, i, t):
        if i == 0:
            x_T = x_T * self.initial_scale
                
        x_t = x_T * self._get_input_scale(i)
        
        t_embed = embed_sinusoidal_position(np.array([t], dtype=float))
        t_embed = torch.tensor(t_embed, device=x_t.device, dtype=torch.float)
        
        with torch.no_grad():
            eps = self.model(x_t, t_embed)
                
        sigma_from = self.sigmas[i]
        sigma_to = self.sigmas[i + 1]
        
        x_T = x_T + eps * (sigma_to - sigma_from)
        
        return x_T