import torch

from torch import nn


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.8, eps=1e-7):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, outputs, targets, logits=False):
        if logits:
            outputs = self.sigmoid(outputs)
        outputs = outputs.clamp(self.eps, 1-self.eps)
        loss = -self.alpha * (1 - outputs)**self.gamma * targets * torch.log(outputs)\
            - (1 - self.alpha) * outputs**self.gamma * (1 - targets) * torch.log(1 - outputs)
        return loss.mean()