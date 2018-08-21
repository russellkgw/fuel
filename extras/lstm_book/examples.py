from numpy import array

# single
# input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# data = array(input)
# print(str(data))
# print(str(data.shape))
# data = data.reshape((1, 10, 1))
# print(str(data))
# print(str(data.shape))

# double
data = array([
[0.1, 1.0],
[0.2, 0.9],
[0.3, 0.8],
[0.4, 0.7],
[0.5, 0.6],
[0.6, 0.5],
[0.7, 0.4],
[0.8, 0.3],
[0.9, 0.2],
[1.0, 0.1]])

print(str(data))
data = data.reshape(1, 10, 2)
print(str(data))
print(data.shape)