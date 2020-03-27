import numpy as np
from collections import defaultdict

def sgd_step(t, w, grad, hyperparams, parameters, proj):
    """
    Toy algo.
    The usual gradient descent.

    :param t: number of that timestep
    :param w: the parameters to update (w for weights)
    :param grad: the gradient vector of the model
    :param hyperparam: dict which contains the hyperparameter specific to this algorithm.
    :param parameters: dict containing the data needed from the previous step
    :return: updated weights and parameters

    Parameters for momentum:
    eta : learning rate
    """

    # Get hyper params, and set them to none if not provided
    hyperparams = defaultdict(lambda: None, hyperparams)

    # get learning rate
    eta = hyperparams['eta']

    # If none is provided set default value
    if eta is None:
        eta = 0.01

    # update weights
    w = w - eta * grad

    if proj:
        w = proj(w)

    # Here parameters from previous are not needed
    return w, parameters
