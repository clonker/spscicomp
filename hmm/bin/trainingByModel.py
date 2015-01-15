
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)


# In[2]:

import hmm.algorithms
import hmm.utility
import hmm.kernel.c


# In[3]:

models = hmm.utility.get_models()


# In[4]:

A, B, pi = models['tbm1']
print A
print B
print pi


# In[5]:

obs = hmm.utility.random_sequence(A, B, pi, 10, kernel=hmm.kernel.python)


# In[6]:

A, B, pi = hmm.utility.generate_startmodel(len(A),len(B[0]))


# In[7]:

A, B, pi, prob, it = hmm.algorithms.baum_welch(
    obs,
    A,
    B,
    pi,
    accuracy=1e-3,
    maxit=1000,
    kernel=hmm.kernel.python,
    dtype=np.float64)


# In[8]:

print A 
print B
print pi


# In[9]:

print it


# In[28]:



