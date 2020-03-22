"""
This file contains the toy training loop to generated video and visualize the algorithms on error surfaces.
"""


import numpy as np
from toys_function import grad_2d, loss_2d
from utils import plot_heatmap, plot_gd, plot_gd2, animate


from optimizers.adam import adam_step
from optimizers.sgd import sgd_step
from optimizers.rmsprop import rms_step
from optimizers.momentum import simple_momentum_step
from optimizers.adagrad import adagrad_step

from matplotlib import pyplot as plt

from copy import copy

def training_loop(epochs, w, step_function, loss_fn, grad_fn, eps=1E-5, **hyperparams):
    """
    :param epochs: Number of step to do
    :param w: initial weights
    :param step_function: the step function to use (the algorithm to use)
    :param loss_fn: the loss function to optimize
    :param grad_fn: the grad function from the loss function
    :param eps: the stopping condition
    :param hyperparams: params for the optimization algorithm
    :return: the init point, all the taken steps and the name of the optimization function
    """
    parameters = None
    t = 1

    init = w

    # We keep every gradient used to draw the path later on
    steps = []

    for e in range(epochs):
        # We take the gradient
        g = grad_fn(w, **hyperparams)

        # Proceed to a step using the parameters and hyperparams
        nw, parameters = step_function(t, w, g, hyperparams, parameters)

        # We compute the steps we made
        steps.append(nw-w)

        # Stopping condition
        if abs(loss_fn(nw)-loss_fn(w)) / (loss_fn(nw)+1E-8) <= eps:
            break

        # Update w
        w=nw

        # Update timestep
        t+=1

    # We plot the error surface and the paths


    return (init, steps,step_function.__name__)


def draw_paths(paths):

    plot_heatmap(loss_2d, -20, 20, -20, 20)

    for init, steps, name in paths:
        plot_gd2(path=(init, steps), label=name)


if __name__=="__main__":
    # We display the background

    init = np.array([-8.,15.])

    all_path = []

    # We optimize the function using our algorithms and we keep the taken path for making a really cool video
    all_path.append(training_loop(300, copy(init), step_function=sgd_step, loss_fn=loss_2d, grad_fn=grad_2d))
    all_path.append(training_loop(300, copy(init), step_function=adam_step, loss_fn=loss_2d, grad_fn=grad_2d))
    all_path.append(training_loop(300, copy(init), step_function=rms_step, loss_fn=loss_2d, grad_fn=grad_2d))
    all_path.append(training_loop(300, copy(init), step_function=simple_momentum_step, loss_fn=loss_2d, grad_fn=grad_2d))
    all_path.append(training_loop(300, copy(init), step_function=adagrad_step, loss_fn=loss_2d, grad_fn=grad_2d))

    draw_paths(all_path)
    plt.savefig("exports/adam-rms-sgd.png")

    # We build the video
    # print(animate(None, all_path, loss_2d))
    # animate("exports/animation-2.mp4", all_path, loss_2d)
