from simulate_2CM import *
from tqdm import tqdm
import numpy as np
from scipy.stats import gamma, multivariate_normal, uniform
from scipy.stats import poisson

def log_likelihood(y, y_sim, sigma):
    # Calculate log-likelihood with scaling to balance acceptance
    return - np.sum((y - y_sim) ** 2) / (2 * sigma**2)

def log_prior(x, sigma):
    # Uniform prior
    return 0

# def log_prior(x, sigma):
#     k1, k2, k3, k4, Vb = x
#     pd_k1 = uniform.pdf(k1, loc=0, scale=1)
#     pd_k2 = uniform.pdf(k2, loc=0, scale=1)
#     pd_k3 = uniform.pdf(k3, loc=0, scale=1)
#     pd_k4 = uniform.pdf(k4, loc=0, scale=1)
#     pd_Vb = uniform.pdf(Vb, loc=0, scale=1)
#     # pd_sigma = gamma.pdf(sigma, a=5e-2, scale=1/0.01)
#     return np.log(pd_k1*pd_k2*pd_k3*pd_k4*pd_Vb)

def log_posterior(y, y_sim, sigma, x):
    # Posterior distribution proportional to the product of prior and likelihood
    return log_prior(x, sigma) + log_likelihood(y, y_sim, sigma)

def update_proposal_std(proposal_std, acceptance_rate, target_acceptance=0.3, iteration=1, m_init=20):
    # Robbins-Monro update for proposal standard deviation
    theta = np.log(proposal_std)
    theta += (acceptance_rate - target_acceptance) / ((iteration + m_init) * target_acceptance * (1 - target_acceptance))
    return np.exp(theta)

def metropolis_hastings(initial_x, y, num_samples, initial_proposal_std, target_acceptance):
    samples = []
    samples.append(initial_x)
    
    # Compute log posterior of the initial state
    _, y_sim = simulate_2CM(initial_x)
    current_log_posterior = log_posterior(y, y_sim, initial_x[-1], initial_x[:-1])
    
    num_accept = 0
    proposal_std = initial_proposal_std
    
    for ite in tqdm(range(1, num_samples + 1)):
        # Generate a proposal state based on the current state
        proposal = np.random.normal(samples[-1], proposal_std, size=samples[-1].shape)

        if np.any(proposal < 0) or np.any(proposal[:-1] > 1.0):
            acceptance_probability = 0
        else:
            # Compute log posterior for the proposal state
            _, y_sim = simulate_2CM(proposal)
            proposal_log_posterior = log_posterior(y, y_sim, proposal[-1], proposal[:-1])
        
            # Compute acceptance prob
            log_acceptance_ratio = proposal_log_posterior - current_log_posterior
            acceptance_probability = min(1, np.exp(log_acceptance_ratio))
    
        # Accept or reject the proposal
        if np.random.rand() < acceptance_probability:
            samples.append(proposal)
            current_log_posterior = proposal_log_posterior
            num_accept += 1
        else:
            samples.append(samples[-1])
            
        # Update proposal standard deviation
        acceptance_rate = num_accept / (ite)
        proposal_std = update_proposal_std(proposal_std, acceptance_rate, target_acceptance, iteration=ite)
    
    print('Acceptance rate: {}'.format(num_accept/num_samples))
    return samples