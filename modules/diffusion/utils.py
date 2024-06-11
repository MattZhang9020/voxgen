import torch

import numpy as np


def embed_sinusoidal_position(timestep, max_period=10000, time_embed_dim=160):
    half = time_embed_dim // 2
    
    log_max_period = np.log(max_period)
    
    arange_vector = np.arange(start=0, stop=half, dtype=np.float32)
    frac_vector = arange_vector / half
    
    log_scaled = -log_max_period * frac_vector
    
    freqs = np.exp(log_scaled)
    
    args = timestep[:, np.newaxis] * freqs[np.newaxis, :]
    
    embedding = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    
    return embedding


def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
