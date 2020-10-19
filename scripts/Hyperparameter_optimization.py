#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('reload_ext', 'autoreload')
import numpy as np
from Implementations import *
from Implementations_helpers import *
from proj1_helpers import *
from Cross_validation import *


# In[ ]:


def Hyperparameter_optimization(y, tx, k_fold, lambdas , gammas, initial_w =1, max_iters = 100, model = 'least_squares'):
    """optimize the hyperameters for correspongding model/algorithm
    lambdas: the range of lambda_
    gammas: the range of gamma
    
    """
    accuracies = np.zeros((len(lambdas), len(gammas)))
    for index_lambda_, lambda_ in enumerate(lambdas):
        for index_gamma, gamma in enumerate(gammas):
            _, accuracy = cross_validation(y, tx, k_fold, lambda_, gamma, initial_w, max_iters, model)
            accuracies[index_lambda_,index_gamma] = accuracy
    optimal_index = np.unravel_index(accuracies.argmax(), accuracies.shape)
    optimal_lambda_, optimal_gamma = lambdas[optimal_index[0]], gammas[optimal_index[1]]
    optimal_accuracy = accuracies.max()
    
    print("Model: {0} \n Optimal accuracy: {1} \n Optimal lambda_: {2} \n Optimal gamma: {3} \n ".format(
          model, optimal_accuracy, optimal_lambda_, optimal_gamma))
    
            
    

