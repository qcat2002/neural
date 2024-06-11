import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 2.7, 1], [1, 4.1, 1], [1, 1.0, 0], [1, 5.2, 1], [1, 2.8, 0]])
print(x)
y = np.array([25, 33, 15, 45, 22])
print(x.shape)
print(y.shape)

iteration = 10000
eta = 0.02
w = np.array([0, 0, 0])
C_W = []
for t in range(iteration):
    grad = 1/5 * (np.matmul(np.matmul(x.T, x), w) - np.matmul(x.T, y))
    w = w - eta * grad
    loss = 1/10 * (np.dot(w.T, np.dot(x.T, np.dot(x, w))) - 2*np.dot(w.T, np.dot(x.T, y)) + np.dot(y.T, y))
    C_W.append(loss)
    if t % 100 == 0:
        print(w)

plt.plot(list(range(iteration)), C_W)
plt.show()