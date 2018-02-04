import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import optimizers
import numpy as np

# program an OR gate.
X = np.array(([[1, 1], [0, 1], [1, 1], [0, 0], [1, 0]]))
y = np.array(([1, 1, 1, 0, 1]))

# create model
model = Sequential()
# 1 neuron, 2 inputs, 1 output
model = Sequential()
model.add(Dense(1, input_dim=2, kernel_initializer="zero", activation='sigmoid'))

# Compile model
sgd = optimizers.SGD(lr=0.1)
model.compile(loss='mean_squared_error' ,optimizer=sgd, metrics=['accuracy'])

# Fit the model for each example (stochastic)
for example, target in zip(X, y):
    #reshape input and output
    example = np.array((example)).reshape(-1,2)
    target = np.array(([target])).reshape(-1,1)

    #forward pass + gradient
    print("\nInput: ", example)
    model.fit(example, target, batch_size=None, epochs=1, verbose=1)
    print("Predicted: ",model.predict(example))
    print("Target: ",target)
    #same behaviour
    # model.train_on_batch(example, target)

