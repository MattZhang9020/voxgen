import torch

import numpy as np

from .utils import get_time_embedding, sigmoid_beta_schedule


class KEulerDiffusionSampler():
    def __init__(self, model, beta_start, beta_end, inference_steps=50, training_steps=1000):        
        self.model = model

        betas = sigmoid_beta_schedule(training_steps, beta_start, beta_end).cpu().numpy()
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        timesteps = np.linspace(training_steps - 1, 0, inference_steps)
        log_sigmas = np.log(sigmas)
        log_sigmas = np.interp(timesteps, range(training_steps), log_sigmas)
        sigmas = np.exp(log_sigmas)
        sigmas = np.append(sigmas, 0)
        
        self.sigmas = sigmas
        
        self.initial_scale = sigmas.max()
    
    def _get_input_scale(self, time_step):
        sigma = self.sigmas[time_step]
        return 1 / (sigma ** 2 + 1) ** 0.5
    
    def __call__(self, x_T, time_step):
        if time_step == 0:
            x_T = x_T * self.initial_scale
                
        t = time_step
        
        x_t = x_T * self._get_input_scale(t)
        
        t_embed = get_time_embedding(np.array([t], dtype=int))
        t_embed = torch.tensor(t_embed, device=x_t.device, dtype=torch.float)
        
        eps = self.model(x_t, t_embed)
                
        sigma_from = self.sigmas[t]
        sigma_to = self.sigmas[t + 1]
        
        x_T = x_T + eps * (sigma_to - sigma_from)
        
        return x_T