#Implement a simple perceptron
import numpy as np

class Perceptron(object):
    def __init__(self,
                 inputSize = 2,
                 learning_rate = 0.1):

        # Define (hyperparameters)
        self.inputSize = inputSize  #number of inputs

        # Weights (parameters)
        # self.W = np.random.randn(self.inputSize)    #weights matrix defined by size of input
        self.W = np.array((0,0))    #weights matrix defined by size of input
        self.eta = learning_rate                    #learning rate

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        # Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def predict(self, X):
        # generated predicted output - Forward pass
        return self.sigmoid(X.dot(self.W))

    def train_batch(self, epochs, X, y):
        # Updates weights according to Percptron Rule. Do not use predict
        for i in range(0,epochs):
            print('\nEpoch %s/%s' % (i + 1, epochs))
            # print('Input: ', X)
            print('Predicted: ', self.predict(X))
            print('%s/%s [==============================] - 0s - loss: %s'
                  % (i, epochs, sum(y - self.predict(X)) ** 2))
            # print("\tInitial weights: ", self.W)
            y_hat = self.predict(X)
            deltaW = (y - y_hat)* self.sigmoidPrime(y_hat)
            self.W = self.W + self.eta* deltaW.dot(X)   #dot product sums all examples
            # # print("\tUpdated weights: ", self.W)
            print('Predicted: ', self.predict(X))
            print('Target: ', y)
        return

    def train_stochastick(self, epochs, X, y):
        for i in range(0, epochs):
            for instance in range(0, X.shape[0]):
                print('\nEpoch %s/%s' % (i+1, epochs))
                print('Input: ', X[instance])
                print('Predicted: ', self.predict(X[instance]))
                print('%s/%s [==============================] - 0s - loss: %s'
                      % (instance, X.shape[0], (y[instance] - self.predict(X[instance]))**2))
                # print("\tInitial weights: ", self.W)
                y_hat = self.predict(X[instance])
                deltaW = (y[instance] - y_hat)* self.sigmoidPrime(y_hat) * X[instance]
                self.W = self.W + self.eta * deltaW
                # print("\tUpdated weights: ", self.W)
                print('Predicted: ', self.predict(X[instance]))
                print('Target: ', y[instance])
        return


# program an OR gate.
X = np.array(([1, 1], [0, 1], [1, 1], [0, 0], [1, 0]))
y = np.array(([1, 1, 1, 0, 1]))

perceptron = Perceptron()
# perceptron.train_stochastick(1, X, y)
perceptron.train_batch(2, X, y)



