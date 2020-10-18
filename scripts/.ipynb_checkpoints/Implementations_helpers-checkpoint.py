# -*- coding: utf-8 -*-
"""Some helper functions used in implementation.py."""
import numpy as np

def compute_loss_LS(y, tx, w):
    """compute the loss of least squares model."""
    e = y - tx.dot(w)  
    loss =  1/2*np.mean(e**2)
    return loss

def compute_gradient_LS(y, tx, w):
    """compute the gradient of least squares model."""
    e = y - tx.dot(w)
    n_sample = y.shape[0]
    gradient = -1/n_sample*tx.T.dot(e)
    return gradient

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def sigmoid(t):
    """apply the sigmoid function on t."""

    return 1 / (1 + np.exp(-t))

def compute_loss_logistic(y,tx,w):
    """compute the loss of logistic regression model."""

    tmp = tx.dot(w)
    loss = y.T.dot(np.log(sigmoid(tmp))) + (1 - y).T.dot(np.log(1-sigmoid(tmp)))
    return loss

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of logistic regression model."""
    tmp = tx.dot(w)
    gradient = tx.T.dot(sigmoid(tmp) - y)
    return gradient

def compute_loss_reg_logistic(y, tx, w, lambda_):
    """compute the loss of regularized logistic regression model."""

    loss = compute_loss_logistic(y, tx, w) + 0.5 * lambda_ * w.T.dot(w)
    return loss

def compute_gradient_reg_logistic(y, tx, w, lambda_):
    """compute the gradient of regularized logistic regression model."""

    gradient = compute_gradient_logistic(y, tx, w) + lambda_ * w
    return gradient



