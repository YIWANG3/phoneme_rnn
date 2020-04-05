import numpy as np

weights = np.array([[0, -1, -1, 1, 1, 1],
                    [-1, 0, 1, -1, -1, -1],
                    [-1, 1, 0, -1, -1, -1],
                    [1, -1, -1, 0, 1, 1],
                    [1, -1, -1, 1, 0, 1],
                    [1, -1, -1, 1, 1, 0]])

vector = np.array([1, -1, -1, 1, 1, 1])
vector2 = np.array([-1, 1, 1, -1, -1, -1])
print(np.dot(weights, vector2))
