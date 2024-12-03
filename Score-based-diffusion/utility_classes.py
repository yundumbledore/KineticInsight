import functools
import numpy as np
import torch
import torch.nn as nn

class GaussianFourierProjection(nn.Module):
  # Gaussian random features for encoding time steps
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        output = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return output

class EmbedY(nn.Module):
  # Encoding label Y
    def __init__(self):
        super().__init__()

    def forward(self, y):
        return y.unsqueeze(-1)

class Dense(nn.Module):
  # A fully connected layer that reshapes outputs to feature maps
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None]
