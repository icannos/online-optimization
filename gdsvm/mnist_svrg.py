import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from svm import hingereg as loss_fn, gradreg as grad_fn, predict

from optimizers.svrg import training_loop
from mnist import plots


if __name__ == "__main__":
    # Step function to use

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
            training_loop(3, grad_fn, loss_fn, np.random.randn(784), x_train, y_train, 512, x_test=x_test,
                          y_test=y_test, lamb=0.5)


