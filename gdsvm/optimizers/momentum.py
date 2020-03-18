import numpy as np
from collections import defaultdict

def simple_momentum_step(t, w, grad, hyperparams, parameters):
    """

    :param t: number of that timestep
    :param w: the parameters to update (w for weights)
    :param grad: the gradient vector of the model
    :param hyperparam: dict which contains the hyperparameter specific to this algorithm.
    :param parameters: dict containing the data needed from the previous step
    :return: updated weights and parameters

    Parameters for momentum:
    eta : learning rate
    gamma: weight of the past (decaying factor for the momentum)
    """

    # Get params, and set None by default if not provided
    hyperparams = defaultdict(lambda: None, hyperparams)
    # Get those we need
    eta, gamma = hyperparams['eta'], hyperparams['gamma']

    # Default parameters if none provided
    if eta is None:
        eta = 0.01
    if gamma is None:
        gamma = 0.5

    if parameters is None:
        parameters = {}
        prev_m = np.zeros_like(w)
    else:
        prev_m = parameters["prev_m"]


    # We use the current grad and the past grads to keep momentum
    m = gamma*prev_m + eta*grad
    # update the weights
    w = w - m
    # set parameters for the next step
    parameters["prev_m"] = m

    return w, parameters