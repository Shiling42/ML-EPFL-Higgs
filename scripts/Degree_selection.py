# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# ## Feature vector extending

# In[2]:


from proj1_helpers import *
from Implementations import *
from Implementations_helpers import *


# In[3]:


## polynomial basis functions
def build_poly(x, degree):
    "polynomial basis functions for input data x, for j=0 up to j=degree."
    "output:new data"

    matrix = np.ones((x.shape[0], 1))
    for j in range(1, degree+1):
        extend = np.power(x, j)
        matrix = np.concatenate((matrix, extend), axis=1)

    return matrix


# ## Degree selection (use ridge regression)

# In[13]:


from Cross_validation import *


# In[16]:


## degree selection
def degree_selection(x,y,degree_range):
    
    for degree in range(degree_range):
        matrix = build_poly(x, degree)
        print('degree=',degree)
        loss,accurancy = cross_validation(y, matrix, k_fold=10, lambda_ = 0.1, gamma = 0.7, initial_w = np.ones(x.shape[1])*1,
                 max_iters = 100, model = 'ridge_regression')

