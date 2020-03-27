import numpy as np
from collections import defaultdict
from svm import hingereg as loss_fn, gradreg as grad_fn, predict
from copy import copy

def svrg_update(w, diffgrad, eta, mu):
    return w - eta * (diffgrad + mu)

def training_loop(epochs, grad_fn, loss_fn, w, x, y, batch_size, x_test=None,
y_test=None, max_steps=None, m=50, eta=0.01, **params):
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

    # On sait quelle taille fait un batch, on compte le nombre de batch que ça donne pour prendre toutes les données
    n_batch = (x.shape[0] // batch_size)

    # utile pour brasser les données à chaque epoch
    idx = np.array([i for i in range(x.shape[0])])

    # Training histories
    loss_records = []
    valloss_records = []
    valaccuracy_records = []
    accuracy_records = []

    parameters = None
    t = 1
    counter = 0

    wtilde = copy(w)

    for e in range(epochs):
        # mélange les indices
        np.random.shuffle(idx)

        # We randomly generates batches
        x_batches = np.array_split(x[idx], n_batch)
        y_batches = np.array_split(y[idx], n_batch)

        # Forme une liste d'éléments de la forme "[(batch_x, batch_y) ... ]"
        batchs = list(zip(x_batches, y_batches))

        # Compute mu
        mu = 0

        # Ce sont les batch qui définissent les fonctions de gradients qu'ont va utiliser,
        # \Psi_i sont définies par les batchi_x, batchi_y
        # C'est grad_fn() qui prend les poids courants, l'entrée en x, les sorties attendues, **parms c'est rien, juste
        # de la technique python
        for i in range(n_batch):
            batch_x = batchs[i][0]
            batch_y = batchs[i][1]

            # Chaque grad \Psi_i
            mu += grad_fn(w, batch_x, batch_y, **params)

        # La moyenne
        mu /= n_batch

        # Fin de compute mu

        for t in range(m):
            it = np.random.randint(n_batch)
            batch_x = batchs[it][0]
            batch_y = batchs[it][1]

            grad_1 = grad_fn(w, batch_x, batch_y, **params)
            grad_2 = grad_fn(wtilde, batch_x, batch_y, **params)

            w = svrg_update(w, grad_1 - grad_2, eta, mu)
            accuracy_records.append((predict(w, batch_x) == batch_y).mean())
            loss_records.append(loss_fn(w, batch_x, batch_y, **params))

        print(accuracy_records[-1])

        # if we provided a test set evaluate the model on it
        if x_test is not None and y_test is not None:
            valloss_records.append(loss_fn(w, x_test, y_test, **params))
            valaccuracy_records.append((predict(w, x_test) == y_test).mean())

        # option 1
        wtilde = copy(w)

    # return the histories
    return loss_records, valloss_records, accuracy_records, valaccuracy_records
