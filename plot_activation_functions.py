import numpy as np
import matplotlib.pyplot as plt


#activations functions to try

def sigmoid(z):
    # Apply sigmoid activation function to scalar, vector, or matrix
    return 1 / (1 + np.exp(-z))


def tanh(z):
    # Apply TANH activation function to scalar vector or matrix
    return (2 / (1 + np.exp(-2 * z))) - 1

def relu(z):
    # Apply RELU activation function to scalar vector or matrix
    return z*(z > 0)

testInput = np.arange(-6,6,0.01)
plt.plot(testInput, relu(testInput), linewidth= 2)
plt.grid(1)
plt.show()