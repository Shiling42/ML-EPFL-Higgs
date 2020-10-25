#!/usr/bin/env python
# coding: utf-8


import numpy as np
from Implementations import *
from Implementations_helpers import *
from proj1_helpers import *
from Cross_validation import *



def Hyperparameter_optimization(y, tx, k_fold, lambdas , gammas, initial_w =1, max_iters = 100, model = 'least_squares'):
    """optimize the hyperameters for correspongding model/algorithm
    lambdas: the range of lambda_
    
         Input:
        - y             = labels
        - tx            = features
        - k_fold        = number of folds
        - lambdas       = the range of L2 prefactor
        - gammas        = the range of learning rate
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
    accuracies = np.zeros((len(lambdas), len(gammas)))
    for index_lambda_, lambda_ in enumerate(lambdas):
        for index_gamma, gamma in enumerate(gammas):
            print('─' * 40)
            print("lambda : {0},  gamma: {1} \n ".format(lambda_, gamma))
            _, accuracy = cross_validation(y, tx, k_fold, lambda_, gamma, initial_w.copy(), max_iters, model)
            accuracies[index_lambda_,index_gamma] = accuracy
    optimal_index = np.unravel_index(accuracies.argmax(), accuracies.shape)
    optimal_lambda_, optimal_gamma = lambdas[optimal_index[0]], gammas[optimal_index[1]]
    optimal_accuracy = accuracies.max()
    print('═' * 40)
    print("*The optimal hyperparameters*:\n Model: {0} \n Optimal accuracy: {1} \n Optimal lambda_: {2} \n Optimal gamma: {3} \n ".format(
          model, optimal_accuracy, optimal_lambda_, optimal_gamma))
    return optimal_accuracy, optimal_lambda_, optimal_gamma
    
            
    

