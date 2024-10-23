from tqdm import tqdm
import numpy as np
from scipy.integrate import odeint

# Define the ODE system
def odefun(x, t, k1, k2, k3, k4):
    # Input function parameters
    beta1 = 264635.097704
    beta2 = 806.522314
    beta3 = 782.270721
    kappa1 = 4.101230
    kappa2 = 0.036614
    kappa3 = 0.037046

    Cf, Cp = x
    Cb = (beta1 * t - beta2 - beta3) * np.exp(-kappa1 * t) + beta2 * np.exp(-kappa2 * t) + beta3 * np.exp(-kappa3 * t)

    dCfdt = -(k2 + k3) * Cf + k4 * Cp + k1 * Cb
    dCpdt = k3 * Cf - k4 * Cp

    return [dCfdt, dCpdt]

# Define the simulation function
def simulate_2CM(param):
    k1, k2, k3, k4, Vb = param

    # Input function parameters
    beta1 = 264635.097704
    beta2 = 806.522314
    beta3 = 782.270721
    kappa1 = 4.101230
    kappa2 = 0.036614
    kappa3 = 0.037046

    # Time span from real data
    tspan = np.array([ 0.13333334,  0.35      ,  0.51666665,  0.68333334,  0.85      ,
        1.0166667 ,  1.1833333 ,  1.4333333 ,  1.7666667 ,  2.1       ,
        2.5166667 ,  3.0166667 ,  3.5166667 ,  4.016667  ,  4.516667  ,
        5.016667  ,  5.766667  ,  6.766667  ,  7.766667  ,  8.766666  ,
        9.766666  , 12.766666  , 17.766666  , 22.766666  , 27.766666  ,
       32.766666  , 37.766666  , 42.766666  , 47.766666  , 52.766666  ])

    # Initial conditions
    x0 = [0, 0]

    # Solve ODE using the exact time points in tspan and precomputed Cb values
    x = odeint(odefun, x0, tspan, args=(k1, k2, k3, k4))

    # Calculate Cb for plot
    Cb = (beta1 * tspan - beta2 - beta3) * np.exp(-kappa1 * tspan) + beta2 * np.exp(-kappa2 * tspan) + beta3 * np.exp(-kappa3 * tspan)

    # Augment state vector with Cb values
    x = np.column_stack((Cb, x))

    # Define observation model
    H = np.array([Vb, 1 - Vb, 1 - Vb])

    # Compute simulated observation
    y = np.dot(x, H)

    return tspan, y

def log_likelihood(y, y_sim, l1):
    lamb = np.log(2) * 1 / 109.8
    delta_t = np.array([1.        , 0.16666667, 0.16666667, 0.16666667, 0.16666667,
       0.16666667, 0.16666667, 0.33333334, 0.33333334, 0.33333334,
       0.5       , 0.5       , 0.5       , 0.5       , 0.5       ,
       0.5       , 1.        , 1.        , 1.        , 1.        ,
       1.        , 5.        , 5.        , 5.        , 5.        ,
       5.        , 5.        , 5.        , 5.        , 5.        ])
    tspan = np.array([ 0.13333334,  0.35      ,  0.51666665,  0.68333334,  0.85      ,
        1.0166667 ,  1.1833333 ,  1.4333333 ,  1.7666667 ,  2.1       ,
        2.5166667 ,  3.0166667 ,  3.5166667 ,  4.016667  ,  4.516667  ,
        5.016667  ,  5.766667  ,  6.766667  ,  7.766667  ,  8.766666  ,
        9.766666  , 12.766666  , 17.766666  , 22.766666  , 27.766666  ,
       32.766666  , 37.766666  , 42.766666  , 47.766666  , 52.766666  ])
    
    sigma_t = l1 * np.sqrt(y_sim*np.exp(-lamb*tspan)/delta_t)*np.exp(lamb*tspan)
    # Calculate log-likelihood with scaling to balance acceptance
    return np.sum(-0.5*np.log(2*np.pi*sigma_t**2) - ((y - y_sim) ** 2)/(2*sigma_t**2))

# def log_prior(x, sigma):
#     # Uniform prior
#     return 0

# def log_prior(x, sigma):
#     k1, k2, k3, k4, Vb = x
#     pd_k1 = uniform.pdf(k1, loc=0, scale=1)
#     pd_k2 = uniform.pdf(k2, loc=0, scale=1)
#     pd_k3 = uniform.pdf(k3, loc=0, scale=1)
#     pd_k4 = uniform.pdf(k4, loc=0, scale=1)
#     pd_Vb = uniform.pdf(Vb, loc=0, scale=1)
#     # pd_sigma = gamma.pdf(sigma, a=5e-2, scale=1/0.01)
#     return np.log(pd_k1*pd_k2*pd_k3*pd_k4*pd_Vb)

def log_posterior(y, y_sim, l1):
    # Posterior distribution proportional to the product of prior and likelihood
    return log_likelihood(y, y_sim, l1)

def update_proposal_std(proposal_std, acceptance_rate, target_acceptance=0.3, iteration=1, m_init=20):
    # Robbins-Monro update for proposal standard deviation
    theta = np.log(proposal_std)
    theta += (acceptance_rate - target_acceptance) / ((iteration + m_init) * target_acceptance * (1 - target_acceptance))
    return np.exp(theta)

def metropolis_hastings(initial_x, y, num_samples, initial_proposal_std, target_acceptance):
    samples = []
    samples.append(initial_x)
    
    # Compute log posterior of the initial state
    _, y_sim = simulate_2CM(initial_x[:-1])
    current_log_posterior = log_posterior(y, y_sim, initial_x[-1])
    
    num_accept = 0
    proposal_std = initial_proposal_std
    
    for ite in tqdm(range(1, num_samples + 1)):
        # Generate a proposal state based on the current state
        proposal = np.random.normal(samples[-1], proposal_std, size=samples[-1].shape)

        if np.any(proposal < 0) or np.any(proposal[:-1] > 1.0):
            acceptance_probability = 0
        else:
            # Compute log posterior for the proposal state
            _, y_sim = simulate_2CM(proposal[:-1])
            proposal_log_posterior = log_posterior(y, y_sim, proposal[-1])
        
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