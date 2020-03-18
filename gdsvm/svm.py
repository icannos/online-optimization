import numpy as np

def hingereg(w, x, y, lamb=None):
    threshold = np.multiply(np.dot(x, w), y)
    cost = np.maximum(1 - threshold, 0)

    return cost.mean() + lamb*np.dot(w.T, w)/2

def gradreg(w, x, y, lamb=None):
    d = w.shape
    threshold = np.multiply(np.dot(x, w), y)
    grad = - np.multiply(x, y.reshape(y.shape[0], 1))

    idx_zeros = (threshold >= 1)

    grad[idx_zeros,:] = np.zeros(d)

    return grad.sum(axis=0)/x.shape[0] + lamb*w

def predict(w, x):
    threshold = np.dot(x, w)
    return 2*(threshold >= 0)-1

