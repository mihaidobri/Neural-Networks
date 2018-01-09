import numpy as np

# add dimensions using [:, np.newaxis] syntax or drop dimensions using np.squeeze
# matrix always have shape (#ofROWS, #ofCOLUMNS)

x = np.array(([1, 1], [0, 1], [1, 0]))
y = np.array([1, 2, 3])
print(y)
print("Shape: ", y.shape)
print("Shape matrix with only one column: ", y[:, np.newaxis].shape)   # a matrix with only one column
print("Shape matrix with only one column: ", y[:, np.newaxis])   # a matrix with only one column
print("Shape matrix with only one row: ", y[np.newaxis, :].shape)   # a matrix with only one row
print("Shape matrix with only one row: ", y[np.newaxis, :])   # a matrix with only one row
print("Shape a 3 dimensional array: ", y[:, np.newaxis, np.newaxis].shape)   # a 3 dimensional array
print("Shape a 3 dimensional array: ", y[:, np.newaxis, np.newaxis])   # a 3 dimensional array
print("Shape original: ", np.squeeze(y[:, np.newaxis, np.newaxis]).shape)   #original

print(x)
print("\nShape: ", x.shape)
print("Shape matrix with only one column: ", x[:, np.newaxis].shape)   # a matrix with only one column
print("Shape matrix with only one row: ", x[np.newaxis, :].shape)   # a matrix with only one row
print("Shape matrix with only one row: ", x[np.newaxis, :])   # a matrix with only one row
print("Shape a 3 dimensional array: ", x[:, np.newaxis, np.newaxis].shape)   # a 3 dimensional array
print("Shape a 3 dimensional array: ", x[:, np.newaxis, np.newaxis])   # a 3 dimensional array
print("Shape original: ", np.squeeze(x[:, np.newaxis, np.newaxis]).shape)   #original


print("\n___\nMultiplicaiton: ", y.dot(x))
print("\n___\nMultiplicaiton: ", y[np.newaxis, :].dot(x))
print("\n___\nMultiplicaiton: ", y[:, np.newaxis, np.newaxis].dot(x[:, np.newaxis, np.newaxis]))

print("\n--------------------------\nTest Shapes:")
# The (length,) array is an array where each element is a number and there are length elements in the array.
# The (length, 1) array is an array which also has length elements, but each element itself is an array with a single element
a = np.array( [[1],[2],[3]] )
print(a.shape)
b = np.array( [1,2,3] )
print(b.shape)