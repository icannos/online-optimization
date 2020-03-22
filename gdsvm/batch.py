
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from svm import hingereg as loss_fn, gradreg as grad_fn, predict

from optimizers.adam import adam_step
from optimizers.sgd import sgd_step
from optimizers.rmsprop import rms_step
from optimizers.momentum import simple_momentum_step
from optimizers.adagrad import adagrad_step

from mnist import training_loop, plots_val

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

    step_functions = [adagrad_step, adam_step, rms_step, simple_momentum_step, sgd_step]
    batch_sizes = [1, 2, 5]

    for batch_size in batch_sizes:
        records = []
        for step_function in step_functions:
            loss_records, valloss_records, accuracy_records, valaccuracy_records = \
                training_loop(1, simple_momentum_step, np.random.randn(784), x_train, y_train, batch_size, x_test=x_test,
                              y_test=y_test, lamb=0.5, max_steps=800)
            plt.figure(1, figsize=(10, 5), dpi=200)
            plots_val(valloss_records, valaccuracy_records,
                  f"exports/mnist-batch{batch_size}-{step_function.__name__}.png", label=step_function.__name__)
            plt.clf()

            records.append((loss_records, valloss_records, accuracy_records, valaccuracy_records))

        plt.clf()
        for k, (loss_records, valloss_records, accuracy_records, valaccuracy_records) in enumerate(records):
            if k == len(records) - 1:
                plots_val(valloss_records, valaccuracy_records,
                      path=f"exports/mnist-batch{batch_size}-comparison.png",
                      label=step_functions[k].__name__)
            else:
                plots_val(valloss_records, valaccuracy_records,
                      label=step_functions[k].__name__,
                      path=-1)


