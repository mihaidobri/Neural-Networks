#Implement a simple perceptron
import numpy as np
import matplotlib.pyplot as plt



class Perceptron(object):
    def __init__(self,
                 inputSize = 2,
                 bias = -1.0,
                 learning_rate = 0.1):

        # Define (hyperparameters)
        self.inputSize = inputSize  #number of inputs
        self.w0 = bias   #threshold (bias)

        # Weights (parameters)
        # self.W = np.random.randn(self.inputSize)    #weights matrix defined by size of input
        self.W = np.array((-1,-1))    #weights matrix defined by size of input
        self.eta = learning_rate                    #learning rate

    def activation(self, X):
        print('Activation: ',X.dot(self.W) + self.w0)
        return X.dot(self.W) + self.w0

    def sgn(self, X):
        print('Prediction: ', [(1.0 if x > 0 else 0) for x in self.activation(X)])
        return [(1.0 if x > 0 else 0) for x in self.activation(X)]

    def train_batch(self, epochs, X, y):
        # Updates weights according to Percptron Rule. Returns deltaW
        print('___')
        for i in range(0,epochs):
            print('\nEpoch: ', i)
            print(self.W)
            self.W = self.W +self.eta*(y - self.sgn(X)).dot(X)
            print(self.W)
        print("-----")
        return self.sgn(X)

    def train_stochastick(self, epochs, X, y):
        print('___')

        for i in range(0, epochs):
            print("Epoch: ",i)
            for batch in range(0, X.shape[0]):
                print('Batch: ', batch)
                print("input: ", X[batch,])
                print(self.W)
                self.W = self.W + self.eta * (y[batch] - self.sgn(X)[batch])*(X[batch,])
                # print(self.W)
                print('\n')
        print('___')
        return self.sgn(X)


# program an AND gate
X = np.array(([1, 1], [0, 1], [1, 1]))
# print(X)
# print(X.shape)
# print("\n-------------\n")
y = np.array(([1, 0, 1]))

perceptron = Perceptron()
predicted = perceptron.sgn(X)
# print(predicted)
# trainec = perceptron.train_batch(16, X, y)
trainec = perceptron.train_stochastick(8, X, y)
print('\nReal: ', y)
print('Predicted: ', trainec)

# plt.plot(X[1:], X[:1] , 'ro')
# plt.grid(1)
# plt.show()



