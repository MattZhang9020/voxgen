import torch


def get_time_embedding(timestep, max_period=10000, time_embed_dim=160):
    half = time_embed_dim // 2
    
    log_max_period = torch.log(torch.tensor(max_period))
    
    arange_vector = torch.arange(start=0, end=half, dtype=torch.float32)
    frac_vector = arange_vector / half
    
    log_scaled = -log_max_period * frac_vector
    
    freqs = torch.exp(log_scaled).to(timestep.device)
    
    args = timestep[:, None] * freqs[None, :]
    
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    return embedding


def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
