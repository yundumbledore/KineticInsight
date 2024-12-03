import torch
import numpy as np
import json

def marginal_prob_std(t, sigma, device):
  # calculate p_{0t}(x(t) | x(0)) std
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma, device):
    # calculate diffusion coeffient
    return torch.tensor(sigma**t, device=device, dtype=torch.float32)

def save_config(model_file_name,
                model_name,
                x_dim,
                y_dim,
                embed_dim,
                lr,
                dropout_prob,
                epoch,
                lr_scheduler_factor,
                lr_scheduler_patience):
    config = {
    "model_name": model_name,
    "x_dim": x_dim,
    "y_dim": y_dim,
    "embed_dim": embed_dim,
    "lr": lr,
    "dropout_prob": dropout_prob,
    "current epoch": epoch,
    "lr_scheduler_factor": lr_scheduler_factor,
    "lr_scheduler_patience": lr_scheduler_patience
}

    with open('{}_config.json'.format(model_file_name), 'w') as f:
        json.dump(config, f, indent=4)

def generate_filtered_tensor(batch_size, x_dim, device):
    num_samples = batch_size * x_dim
    lower_bound = -1.96
    upper_bound = 1.96
    filtered_numbers = torch.tensor([], device=device)

    while filtered_numbers.numel() < num_samples:
        random_numbers = torch.randn(num_samples, device=device)
        filtered_additional = random_numbers[(random_numbers >= lower_bound) & (random_numbers <= upper_bound)]
        filtered_numbers = torch.cat((filtered_numbers, filtered_additional))

    # Reshape the filtered numbers to match the desired format
    filtered_numbers = filtered_numbers[:num_samples].view(batch_size, 1, x_dim)
    return filtered_numbers

def remove_outliers_iqr(data):
    data_array = np.array(data)
    data_array = data_array[data_array > 0]
    data_array = data_array[~np.isnan(data_array)]

    # Calculate the first (Q1) and third (Q3) quartiles
    Q1 = np.percentile(data_array, 25)
    Q3 = np.percentile(data_array, 75)

    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Determine the bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the data to exclude outliers
    filtered_data = [x for x in data_array if lower_bound <= x <= upper_bound]

    return filtered_data
