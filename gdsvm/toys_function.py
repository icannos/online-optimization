"""
This file contains some toy function to optimize, to test and visualize our algorithms
"""

import numpy as np
from matplotlib import pyplot as plt

def grad1(x, *args, **kwargs):
    """
    Get grad for loss 1
    :param x: point
    :return:
    """
    return 2 * x - 4


def loss1(x, *args, **kwargs):
    """
    Simple quadratique 1D loss
    :param x:
    """
    return x ** 2 - 4 * x + 2


def loss_2d(x):
    """
    An error surface with 2 local optima one better than the other.
    :param x:
    :return:
    """
    y1 = np.array([-10, 0])
    y2 = np.array([10, 0])
    return np.linalg.norm(x - y1) * (np.linalg.norm(x - y2) + 2)


def grad_2d(x):
    """
    Grad for loss 2
    :param x:
    :return:
    """
    y1 = np.array([-10, 0])
    y2 = np.array([10, 0])

    return (((x - y1) / np.sqrt(np.linalg.norm(x - y1))) * (np.linalg.norm(x - y2) + 2) \
            + ((x - y2) / np.sqrt(np.linalg.norm(x - y2))) * np.linalg.norm(x - y1))
