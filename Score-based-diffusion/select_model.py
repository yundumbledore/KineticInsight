from models import *

def model_initializer(model_name, x_dim, y_dim, marginal_prob_std_fn, embed_dim):
    if model_name == 'OneDUnet':
        return OneDUnet(x_dim, y_dim, marginal_prob_std_fn, embed_dim)
