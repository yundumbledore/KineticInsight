import torch
import random

## Define loss function
def loss_fn(model, x, y, dropout_prob, marginal_prob_std, eps=1e-8):
    # eps: A tolerance value for numerical stability.
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None]
    # Classifier free guidance-Randomly drop the conditioning information
    if random.random() < dropout_prob:
        y = torch.ones_like(y) * 10  # Replace with dummy value
    score = model(perturbed_x, y, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None] + z)**2, dim=(1,2)))
    return loss
