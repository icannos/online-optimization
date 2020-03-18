import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from svm import hingereg as loss_fn, gradreg as grad_fn, predict

from optimizers.adam import adam_step
from optimizers.sgd import sgd_step
from optimizers.rmsprop import rms_step
from optimizers.momentum import simple_momentum_step
from optimizers.adagrad import adagrad_step


def plots(loss_records, valloss_records, accuracy_records, valaccuracy_records, path=None):
    """
    Utils function to plot the mnist train history
    :param path: where to dump the images (if None, it will be displayed)
    :param loss_records: list of loss
    :param valloss_records: list of validation loss
    :param accuracy_records: list of accuracy scores
    :param valaccuracy_records: list of validation accuracy scores
    :return:
    """
    plt.figure(1, figsize=(20, 10), dpi=200)
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(2, 2, 1)
    plt.title("Train loss")
    plt.plot([i for i in range(len(loss_records))], loss_records)

    plt.subplot(2, 2, 2)
    plt.title("Train accuracy")
    plt.plot([i for i in range(len(accuracy_records))], accuracy_records)

    plt.subplot(2, 2, 3)
    plt.title("Validation loss")
    plt.plot([i for i in range(len(valloss_records))], valloss_records)

    plt.subplot(2, 2, 4)
    plt.title("Validation accuracy")
    plt.plot([i for i in range(len(valaccuracy_records))], valaccuracy_records)

    if path:
        plt.savefig(path)
    else:
        plt.show()


def training_loop(epochs, step_function, w, x, y, batch_size, x_test=None, y_test=None, **params):
    """

    :param epochs: number of epoch to do
    :param step_function: optimization algorithm
    :param w: svm weights
    :param x: x input (here mnist images)
    :param y: target classes
    :param batch_size: size of a minibatch
    :param x_test: x test
    :param y_test: y test
    :param params: parameters for the optimization algorithms
    :return:
    """
    n_batch = (x.shape[0] // batch_size)

    idx = np.array([i for i in range(x.shape[0])])

    # Training histories
    loss_records = []
    valloss_records = []
    valaccuracy_records = []
    accuracy_records = []

    parameters = None
    t = 1

    for e in range(epochs):
        np.random.shuffle(idx)

        # We randomly generates batches
        x_batches = np.array_split(x[idx], n_batch)
        y_batches = np.array_split(y[idx], n_batch)

        # We iterate over the batches
        for batch_x, batch_y in zip(x_batches, y_batches):
            # get gradient
            g = grad_fn(w, batch_x, batch_y, **params)

            # make an optimization step
            w, parameters = step_function(t, w, g, params, parameters)

            # store scores for plots
            loss_records.append(loss_fn(w, batch_x, batch_y, **params))

            # if we provided a test set evaluate the model on it
            if x_test is not None and y_test is not None:
                valloss_records.append(loss_fn(w, x_test, y_test, **params))
                valaccuracy_records.append((predict(w, x_test) == y_test).mean())

            accuracy_records.append((predict(w, batch_x) == batch_y).mean())

    # return the histories
    return loss_records, valloss_records, accuracy_records, valaccuracy_records


if __name__ == "__main__":
    # Step function to use
    step_function = adagrad_step

    # Open and format data

    data_train = pd.read_csv("data/mnist_train.csv").values
    data_test = pd.read_csv("data/mnist_test.csv").values

    y_train = data_train[:, 0]
    x_train = (data_train[:20000, 1:] / 255)

    y_test = data_test[:5000, 0]
    x_test = data_test[:5000, 1:] / 255

    y_train = 2 * np.array(y_train == 0) - 1  # classification 0/autres
    y_test = 2 * np.array(y_test == 0) - 1  # classification 0/autres

    loss_records, valloss_records, accuracy_records, valaccuracy_records = \
        training_loop(2, simple_momentum_step, np.random.randn(784), x_train, y_train, 512, x_test=x_test,
                      y_test=y_test, lamb=0)

    # store the plots
    plots(f"exports/{step_function.__name__}.png", loss_records, valloss_records, accuracy_records, valaccuracy_records)
