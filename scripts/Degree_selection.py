# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
from Cross_validation import *
from proj1_helpers import *
from Implementations import *
from Implementations_helpers import *


## Feature vector extending
## Polynomial basis functions
def build_poly(x, degree):
    """
    Build up polynomial basis functions for input data x, for j=0 up to j=degree.

    Input:
        - x         = the input data (features)
        - degree    = the desired degree 
    
    Output:
        - extended feature = [tx^1,...,tx^degree]
    """
    matrix = np.ones((x.shape[0], 1))
    for j in range(1, degree+1):
        extend = np.power(x, j)
        matrix = np.concatenate((matrix, extend), axis=1)

    return matrix


## Degree selection (use ridge regression)
def degree_selection(x,y,degree_range):
    """
    Select the optimal degree for best accuracy

    Input:
        - x             = feature
        - y             = label
        - degree_range  = the range of degree for selection

    Output:
        - print accuracy for all input degrees

    """
    for degree in range(degree_range):
        matrix = build_poly(x, degree)
        print('degree=',degree)
        loss,accurancy = cross_validation(y, matrix, k_fold=10, lambda_ = 0.1, gamma = 0.7, initial_w = np.ones(x.shape[1])*1,
                 max_iters = 100, model = 'ridge_regression')

