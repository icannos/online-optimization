import numpy as np
from collections import defaultdict


def adabound_step(t, w, grad, hyperparams, parameters, proj=None):
    """

    :param t: number of that timestep
    :param w: the parameters to update (w for weights)
    :param grad: the gradient vector of the model
    :param hyperparam: dict which contains the hyperparameter specific to this algorithm.
    :param parameters: dict containing the data needed from the previous step
    :param proj: projection function that can be used to enforce a feasible region
    :return: updated weights and parameters

    Parameters for adabound:
    eps : float, it is used to avoid division by 0
    eta : learning rate
    beta1 : Decaying rate for the running mean
    beta2 : Decaying rate for the running deviation

    lower_bound_fn : function t -> bound
    upper_bound_fn : function t -> bound
    """

    # Retrieve the hyperparam and set to default None if not set
    hyperparams = defaultdict(lambda: None, hyperparams)

    beta1, beta2, eta, eps = hyperparams['beta1'], hyperparams['beta2'], hyperparams['eta'], hyperparams['eps']
    lower_bound_fn = hyperparams["lower_bound_fn"]
    upper_bound_fn = hyperparams["upper_bound_fn"]

    # Default parameters
    if beta1 is None:
        beta1 = 0.9
    if beta2 is None:
        beta2 = 0.999
    if eta is None:
        eta = 1
    if eps is None:
        eps = 1E-8

    if upper_bound_fn is None:
        upper_bound_fn = lambda t: 1 * (1 - np.exp(-0.1*t))
    if lower_bound_fn is None:
        lower_bound_fn = lambda t: 1 * (100*np.exp(0.1 * t) + 1)

    # Get needed parameters from last step if they exist
    if parameters is None:
        parameters = {}
        prev_m = np.zeros_like(w)
        prev_v = np.zeros_like(w)
    else:
        prev_m = parameters["prev_m"]
        prev_v = parameters["prev_v"]

    # Computing adam mean and variance
    m = beta1 * prev_m + (1 - beta1) * grad
    v = beta2 * prev_v + (1 - beta2) * np.power(grad, 2)

    # We correct them to avoid biais to 0
    mchap = m / (1 - np.power(beta1, t))
    vchap = v / (1 - np.power(beta2, t))

    # We compute the usual step size of adam
    eta_adam = eta / (np.sqrt(vchap) + eps)

    # We apply clippling using the bounding functions
    eta_t = np.clip(eta_adam, lower_bound_fn(t), upper_bound_fn(t)) / np.sqrt(t)

    # update weights
    w = w - eta_t * mchap

    if proj:
        w = proj


    # update params for next step
    parameters["prev_m"] = m
    parameters["prev_v"] = v

    return w, parameters
