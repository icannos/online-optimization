import numpy as np
from collections import defaultdict

def rms_step(t, w, grad, hyperparams, parameters):
    """

    :param t: number of that timestep
    :param w: the parameters to update (w for weights)
    :param grad: the gradient vector of the model
    :param hyperparam: dict which contains the hyperparameter specific to this algorithm.
    :param parameters: dict containing the data needed from the previous step
    :return: updated weights and parameters

    Parameters for RMSProp:
    eps : float, it is used to avoid division by 0
    eta : learning rate
    beta1 : Decaying rate for the running mean
    """
    # Retrieve the hyperparam and set to default None if not set
    hyperparams = defaultdict(lambda: None, hyperparams)
    beta1, eta, eps = hyperparams['beta1'], hyperparams['eta'], hyperparams['eps']

    # Default parameters if none are given
    if beta1 is None:
        beta1 = 0.95
    if eta is None:
        eta = 1
    if eps is None:
        eps = 1E-8

    # Get params from previous step
    if parameters is None:
        parameters = {}
        prev_v = np.zeros_like(w)
    else:
        prev_v = parameters["prev_v"]

    # We compute the running uncenterd variance
    v = beta1 * prev_v + (1-beta1)*np.power(grad, 2)

    # Update of the weights
    w = w - (eta / (np.sqrt(v)+eps)) * grad

    # Update the parameters for next step
    parameters["prev_v"] = v

    return w, parameters