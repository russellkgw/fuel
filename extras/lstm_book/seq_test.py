from numpy import array

x = array([[1.0, 2.0], [1.1, 2.0], [0.5, 0.6],
           [1.0, 2.0], [1.1, 2.0], [0.5, 0.6],
           [1.0, 2.0], [1.1, 2.0], [0.5, 0.6]])


y = array([3.0, 3.0, 1.1, 3.0, 3.0, 1.1, 3.0, 3.0, 1.1])

# Shape
print('X shape: ' + str(x.shape))
print('Y shape: ' + str(y.shape))

# Re Shape
x_reshape = x.reshape(1, 9, 2)
y_reshape = y.reshape(1, 9, 1)
print('X shape: ' + str(x_reshape.shape))
print('Y shape: ' + str(y_reshape.shape))
