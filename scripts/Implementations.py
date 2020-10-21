#!/usr/bin/env python
# coding: utf-8

import numpy as np
from Implementations_helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Implementations of linear regression using gradient descent"""
    w = initial_w
    ws = [w]
    loss = compute_loss_LS(y, tx, w)
    losses = [loss]
    for n_iter in range(max_iters):
        gradient = compute_gradient_LS(y, tx, w)
        w -= gamma * gradient
        loss = compute_loss_LS(y, tx, w)
        ws.append(w)
        losses.append(loss)
#        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
#              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses[-1], ws[-1]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Implementations of linear regression using stochastic gradient descent"""
    w = initial_w
    ws = [w]
    loss = compute_loss_LS(y, tx, w)
    losses = [loss]
    for iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            gradient = compute_gradient_LS(y_batch, tx_batch, w)
            w -= gamma * gradient
            loss = compute_loss_LS(y_batch, tx_batch, w)
            ws.append(w)
            losses.append(loss)
     #       print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
     #             bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses[-1], ws[-1]

def least_squares(y, tx):
    """Implementations of least squares regression using normal equation"""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    loss =compute_loss_LS(y,tx,w)
    return loss, w

def ridge_regression(y, tx, lambda_):
    """Implementations of ridge regression using normal equation"""
    N = tx.shape[0]
    a = tx.T.dot(tx) + 2 * N * lambda_ * np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss_LS(y, tx, w)  
    return loss, w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Implementations of logistic regression using gradient descent"""
    w = initial_w
    #print(w)
    loss = compute_loss_logistic(y, tx, w)
    losses = [loss]
    ws = [w]
    for iter in range(max_iters):
        gradient = compute_gradient_logistic(y, tx, w)
        w -= gamma * gradient
        ws.append(w)
        loss = compute_loss_logistic(y, tx, w)
        if iter % int(max_iters/10) == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        losses.append(loss)
    return losses[-1],ws[-1]

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Implementations of regularized logistic regression using gradient descent"""
    w = initial_w
    ws = [w]
    loss = compute_loss_logistic(y, tx, w)
    losses = [loss]
    for iter in range(max_iters):
        gradient = compute_gradient_reg_logistic(y, tx, w, lambda_)
        w -= gamma * gradient
        loss = compute_loss_logistic(y, tx, w)
        if iter % int(max_iters/10) == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        losses.append(loss)
        ws.append(w)
    return losses[-1], ws[-1]





