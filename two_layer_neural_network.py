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

# Input data, transpose, layer 1, layer 2, biases
def train(x, t, V, W, dv, dw):
    # Forward propogation -- matrix multiply + biases
    A = np.dot(x, V) + bv
    Z = np.tanh(A)

    B = np.dot(Z, W) + dw
    Y = sigmoid(B)

    # Backward propogation
    Ew = Y - t
    Ev = tanh_prime(A) + np.dot(W, Ew)

    # Predict our loss
    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)

    # cross entropy
    loss = -np.mean(t * np.log(Y)) + (1 - t) * np.log(1 - Y)

    return loss, (dV, dW, Ev, Ew)

def predict(x, V, W, dv, dw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    return (sigmoid(B) > 0.5).astype(int)

# Create layers
V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V, W, bv, bw]

# Generate our data
X = np.random.binomial(1, 0.5, (n_sample, n_in))
T = X ^ 1

# Training
for epoch in range(100):
    err = []
    upd = [0] * len(params)

    t0 = time.clock()

    # For each data point, update weights
    for i in range(X.shape[0]):
        loss, grad = train(X[i], T[i], *params)
        # Update loss
        for j in range(len(params)):
            params[j] -= upd[j]
        for j in range(len(params)):
            upd[j] = learning_rate * grad[j] + momentum * upd[j]
        err.append(loss)

    print("Epoch: {0}, Loss: {1}, Time: {2}".format(epoch, np.mean(err), time.clock() - t0))


# Try to predict something
x = np.random.binomial(1, 0.5, n_in)
print("XOR Prediction")
print(x)
print(predict(x, *params))
