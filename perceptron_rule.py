#Implement a simple perceptron
import numpy as np

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

    #generated predicted output - Forward pass
    def predict(self, X):
        activation  = X.dot(self.W)
        # print("Activation: ",activation)
        # thresholded outp®ut
        return [(1.0 if i + self.w0 >= 0 else 0) for i in activation]

    def train_batch(self, epochs, X, y):
        # Updates weights according to Percptron Rule. Do not use predict
        print('___')
        for i in range(0,epochs):
            output = X.dot(self.W)
            print('\nEpoch: ', i)
            print("Initial weights: ",self.W)
            self.W = self.W +self.eta*(y - self.predict(X)).dot(X)
            print("Updated weights: ",self.W)
        print("-----")
        return

    def train_stochastick(self, epochs, X, y):
        print('___')

        for i in range(0, epochs):
            print("Epoch: ",i)
            for instance in range(0, X.shape[0]):
                print('\tInstance: ', instance)
                print("\t\tinput: ", X[instance,])
                print("\t\tWeights before: ",self.W)
                print("\t\tyhat: ",self.predict(X)[instance])
                print("\t\ty: ",y[instance])
                output = X[instance].dot(self.W)
                self.W = self.W + self.eta * (y[instance] - output)*(X[instance,])
                print('\t\tWeights after: ',self.W)
                print('\n')
        print('___')
        return


# program an OR gate.
X = np.array(([1, 1], [0, 1], [1, 1], [0, 0], [1, 0]))
y = np.array(([1, 1, 1, 0, 1]))

perceptron = Perceptron()
y_hat = perceptron.predict(X)
print("Predicted: ",y_hat)
print('\nTarget: ', y)
perceptron.train_batch(6, X, y)
y_hat = perceptron.predict(X)
print("Predicted: ",y_hat)
print('\nTarget: ', y)



