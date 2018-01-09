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
        self.W = np.array((0,0))    #weights matrix defined by size of input
        self.eta = learning_rate                    #learning rate

    def activation(self, X):
        return X.dot(self.W)

    def sgn(self, X):
        #thresholded output
        return [(1.0 if x+ self.w0 > 0 else 0) for x in self.activation(X)]

    def train_batch(self, epochs, X, y):
        # Updates weights according to Percptron Rule. Returns deltaW
        print('___')
        for i in range(0,epochs):
            print('\nEpoch: ', i)
            print("Initial weights: ",self.W)
            self.W = self.W +self.eta*(y - self.sgn(X)).dot(X)
            print("Weight: ", self.W)
            print("Updated weights: ",self.W)
        print("-----")
        return self.sgn(X)

    def train_stochastick(self, epochs, X, y):
        print('___')

        for i in range(0, epochs):
            print("Epoch: ",i)
            for batch in range(0, X.shape[0]):
                print('\tBatch: ', batch)
                print("\t\tinput: ", X[batch,])
                print("\t\tWeights before: ",self.W)
                print("\t\tyhat: ", self.activation(X)[batch])
                print("\t\ty: ",y[batch])
                self.W = self.W + self.eta * (y[batch] - self.activation(X)[batch])*(X[batch,])
                print('\t\tWeights after: ',self.W)
                print('\n')
        print('___')
        return self.activation(X)


# program an AND gate.
X = np.array(([1, 1], [0, 1], [1, 1]))
y = np.array(([1, 0, 1]))

perceptron = Perceptron()
predicted = perceptron.sgn(X)
# trainec = perceptron.train_batch(16, X, y)
trainec = perceptron.train_stochastick(10, X, y)
print('\nReal: ', y)
print('Predicted: ', trainec)



line1, = plt.plot(y, label="Line 1", linestyle='--')
line2, = plt.plot([3,2,1], label="Line 2", linewidth=4)

# Create a legend for the first line.
first_legend = plt.legend(handles=[line1], loc=1)

# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)

# Create another legend for the second line.
plt.legend(handles=[line2], loc=4)

plt.show()

