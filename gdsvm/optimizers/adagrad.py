import numpy as np
from collections import defaultdict

def adagrad_step(t, w, grad, hyperparam, parameters, proj=None):
    """

    :param t: number of that timestep
    :param w: the parameters to update (w for weights)
    :param grad: the gradient vector of the model
    :param hyperparam: dict which contains the hyperparameter specific to this algorithm.
    :param parameters: dict containing the data needed from the previous step
    :param proj: projection function that can be used to enforce a feasible region
    :return: updated weights and parameters

    Parameters for adagrad:
    eps : float, it is used to avoid division by 0
    eta : learning rate
    """

    # Retrieve the hyperparam and set to default None if not set
    hyperparam = defaultdict(lambda: None, hyperparam)

    # We keep only the ones useful for that method
    eta, eps = hyperparam['eta'], hyperparam['eps']

    # If none we set default values
    if eta is None:
        eta = 1
    if eps is None:
        eps = 1E-8

    # If we have no parameters from previous step (ie first step), we initialize those parameters
    if parameters is None:
        parameters = {}
        prev_G = np.zeros_like(w)
    # Else we retrieve what we want from the previous step
    else:
        prev_G = parameters["prev_G"]

    # We compute the running mean of the uncentered deviation
    G = prev_G + np.power(grad, 2)

    # We update the weights according
    w = w - (eta / (np.sqrt(G)+eps)) * grad

    if proj:
        w = proj(w)

    # Then set the parameters for next step
    parameters["prev_G"] = G

    # We return both the new weigths and the updated parameters
    return w, parameters