# Two layer Neural Network

import numpy as np
import time

# Variables
n_hidden = 10
n_in = 10

# Outputs
n_out = 10

# Sample data
n_sample = 300

# Hyperparmeters
learning_rate = 0.01
momentum = 0.9

# Non deterministic seeding
np.random.seed(0)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh_prime(x):
    return 1 - (np.tanh(x) ** 2)
