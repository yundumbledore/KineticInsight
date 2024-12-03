import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the ODE system
def odefun(x, t, k1, k2, k3, k4, beta1, beta2, beta3, kappa1, kappa2, kappa3):
    Cf, Cp = x
    Cb = (beta1 * t - beta2 - beta3) * np.exp(-kappa1 * t) + beta2 * np.exp(-kappa2 * t) + beta3 * np.exp(-kappa3 * t)

    dCfdt = -(k2 + k3) * Cf + k4 * Cp + k1 * Cb
    dCpdt = k3 * Cf - k4 * Cp

    return [dCfdt, dCpdt]

# Define the simulation function
def simulate_2CM(param):
    k1, k2, k3, k4, Vb, sigma = param

    # Input function parameters
    beta1 = 12
    beta2 = 1.8
    beta3 = 0.45
    kappa1 = 4
    kappa2 = 0.5
    kappa3 = 0.008

    # Initial conditions
    x0 = [0, 0]

    # Time span
    fs = 1 / 60  # sampling rate
    t_simulation = 61  # minutes
    tspan = np.linspace(0, t_simulation, int(t_simulation * 60 * fs))

    # Solve ODE
    x = odeint(odefun, x0, tspan, args=(k1, k2, k3, k4, beta1, beta2, beta3, kappa1, kappa2, kappa3))

    # Calculate Cb for plot
    Cb = (beta1 * tspan - beta2 - beta3) * np.exp(-kappa1 * tspan) + beta2 * np.exp(-kappa2 * tspan) + beta3 * np.exp(-kappa3 * tspan)

    # Augment state vector
    x = np.column_stack((Cb, x))

    # Define observation model
    H = np.array([Vb, 1 - Vb, 1 - Vb])

    # Compute simulated observation
    y = np.dot(x, H)

    return tspan, y