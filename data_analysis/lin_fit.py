import numpy as np

x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
# x = np.array([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1],  [4.0, 4.1],  [5.0, 51.]])

y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])

z = np.polyfit(x, y, 3)

print(str(z))