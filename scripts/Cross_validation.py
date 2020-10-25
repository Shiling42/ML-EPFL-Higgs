#!/usr/bin/env python
# coding: utf-8




import numpy as np
from Implementations import *
from proj1_helpers import *

def build_k_indices(y, k_fold, seed):
    """
    build k indices for k-fold.
    
    Input:
        - y         = data for k-fold
        - k_fold    = the number of folds
        - seed      = seed for random number geneartor

    Output:
        - an k-fold data
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def compute_accuracy(tx_test, y_test, w, model):
    """
    compute the accuracy rate of corresponding model
    
    Input: 
        - tx_test   = samples' features for test
        - y_test    = labels for test
        - w         = weights of the input model
        - model     = 'least_squares'
                      'least-squares_GD'
                      'least_squares_SGD'
                      'ridge_regression'
                      'logistic'ragression'
                      'reg_logistic_regression_'
    Output:
        - accuracy  = the accuracy of the input model
    """
    accuracy = 1-np.sum(np.abs(y_test-predict_labels(w, tx_test,model)))/len(tx_test)
    
    return accuracy

def cross_validation(y, tx, k_fold, lambda_ = 0.1, gamma = 0.7, initial_w =1, max_iters = 100, model = 'least_squares'):
    """
     k-fold cross validation
     return the accuracy and loss of corresponding model
     
     Input:
        - y             = labels
        - tx            = features
        - k_fold        = number of folds
        - lambda_       = prefactor of L2 norm
        - gamma         = learning rate (step size)
        - initial_w     = initial weights
        - max_iters     = maximum iterations
        - model         = 'least_squares'
                          'least-squares_GD'
                          'least_squares_SGD'
                          'ridge_regression'
                          'logistic_ragression'
                          'reg_logistic_regression_'
    
    Output:
        - loss          = loss of the final trained model
        - accuracy      = accuracy of the final trained model
    """
    seed = 2
    k_indices = build_k_indices(y, k_fold, seed)
    loss = 0
    accuracy = 0
    for i in range(k_fold):
        tx_test = tx[k_indices[i]]
        y_test = y[k_indices[i]]
        tx_train = np.delete(tx, k_indices[i], axis = 0)
        y_train = np.delete(y, k_indices[i], axis = 0)
        if model == 'least_squares_GD':
            loss_temp, w_temp = least_squares_GD(y_train, tx_train, initial_w.copy(), max_iters, gamma)
            loss = loss + loss_temp
            accuracy_tmp= compute_accuracy(tx_test, y_test, w_temp, model)
            accuracy = accuracy + accuracy_tmp
        elif model == 'least_squares_SGD':
            loss_temp, w_temp = least_squares_SGD(y_train, tx_train, initial_w.copy(), max_iters, gamma)
            loss = loss + loss_temp
            accuracy_tmp = compute_accuracy(tx_test, y_test, w_temp, model)
            accuracy = accuracy + accuracy_tmp
        elif model == 'least_squares':
            loss_temp, w_temp = least_squares(y_train, tx_train)
            loss = loss + loss_temp
            accuracy_tmp= compute_accuracy(tx_test, y_test, w_temp, model)
            accuracy = accuracy + accuracy_tmp
        elif model == 'ridge_regression':
            loss_temp, w_temp = ridge_regression(y_train, tx_train, lambda_)
            loss = loss + loss_temp
            accuracy_tmp= compute_accuracy(tx_test, y_test, w_temp, model)
            accuracy = accuracy + accuracy_tmp
        elif model == 'logistic_regression':
            loss_temp, w_temp = logistic_regression(y_train, tx_train, initial_w.copy(), max_iters, gamma)
            loss = loss + loss_temp
            accuracy_tmp= compute_accuracy(tx_test, y_test, w_temp, model)
            accuracy = accuracy + accuracy_tmp
        elif model == 'reg_logistic_regression':
            loss_temp, w_temp = reg_logistic_regression(y_train, tx_train, lambda_, initial_w.copy(), max_iters, gamma)
            loss = loss + loss_temp
            accuracy_tmp= compute_accuracy(tx_test, y_test, w_temp, model)
            accuracy = accuracy + accuracy_tmp
        else:
            raise ValueError("Unknown model! Please check it!")
            
    loss = np.squeeze(loss / k_fold)
    accuracy = accuracy / k_fold
    print(' Model', model,'\n Loss:', loss, '\n Accuracy:', accuracy)
    return loss, accuracy
    