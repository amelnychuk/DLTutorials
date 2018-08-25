import numpy as np

import matplotlib.pyplot as plt

Nclass = 500

X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])

Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

D = 2
M = 3
K = 3

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)

W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def relu(Z):
    return np.maximum(Z, 0, axis=1, keepdims=True)

def softmax(Z):
    expZ = np.exp(Z)
    return expZ / expZ.sum(axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    Z = relu(X.dot(W1) + b1)
    return relu(Z.dot(W2)+ b2)

def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) + b1))
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y

def classification_rate(Y, P):
    return np.sum(Y == np.round(P)) / len(Y)


P_Y_given_X = forward(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)

assert(len(P) == len(Y))

class_rate = classification_rate(Y, P)
print(f"Classification rate: {class_rate}")
