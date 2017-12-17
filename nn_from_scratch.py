import numpy as np



class Neural_Network(object):
    def __init__(self):
        # Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def sigmoidPrime(self, z):
        # Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat) ** 2)
        return J

    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def forward_sigmoid(self, X):
        # Propagate inputs though network using Sigmoid
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def forward_tanh(self, X):
        # Propagate inputs though network using TANH
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.tanh(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.tanh(self.z3)
        return yHat

    def forward_relu(self, X):
        # Propagate inputs though network using RELU
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.relu(self.z3)
        return yHat

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        # Apply TANH activation function to scalar vector or matrix
        return (2/ (1 + np.exp(-2*z)))-1

    def relu(self, z):
        # Apply RELU activation function to scalar vector or matrix
        return z * (z > 0)

# Propagate inputs though network


#generate data
print("Regular data:\n")
print("Input:")
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
print(X.shape)
print(X,"\n")
print("Output:")
y = np.array(([75], [82], [93]), dtype=float)
print(y.shape)
print(y,"\n")

#scale the data
print("\nScaled data:\n")
X = X/np.amax(X, axis=0)
print(X.shape)
print(X,"\n")

y = y/100 #Max test score is 100
print(y.shape)
print(y,"\n")

print("Predictions:\n")
NN = Neural_Network()
y_hat = NN.forward_sigmoid(X)
print(y_hat,"\n")

y_hat = NN.forward_tanh(X)
print(y_hat,"\n")

y_hat = NN.forward_relu(X)
print(y_hat,"\n")

print("\nProgram finsihed succesfully!")