import torch

def get_time_embedding(timestep, dim, max_period=10000):
    half = dim // 2
    
    log_max_period = torch.log(torch.tensor(max_period))
    arange_vector = torch.arange(start=0, end=half, dtype=torch.float32)
    frac_vector = arange_vector / half
    log_scaled = -log_max_period * frac_vector
    freqs = torch.exp(log_scaled)
    
    args = timestep[:, None] * freqs[None, :]
    
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    return embedding