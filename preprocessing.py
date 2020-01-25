#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing Library
# 
# 1. Environment Setup: import required libraries
# 2. Calculating Features Statistic Data
# 3. Preprocessing Data
# 4. Piping Preprocessing Data
# 

# ## 1. Environment Setup: import required library
# 

# In[ ]:


import utility

# install library
import numpy as np


# ## 2. Calculating Feature Statistic Data
# 
# We will write a function to calculate the training and validating data. The mean, maximum, standard deviation, and variance are returned at the end of the function.
# 

# In[ ]:


def get_feat_stat(arr):
    arr = [ f.reshape((-1, f.shape[-1])) for f in arr ]
    arr = np.concatenate(arr)

    return {
            'max': np.max(arr, axis=0),
            'mean': np.mean(arr, axis=0, dtype=np.float128),
            'stdev': np.nanstd(arr, axis=0, dtype=np.float128),
            'var': np.nanvar(arr, axis=0, dtype=np.float128),
            }


# In[ ]:


if __name__ == "__main__":

    A = np.arange(2*3*6).reshape((2,3,6))

    stat = get_feat_stat([A])

    # stat = get_feat_stat(X_train_orig)
    print("create a simple matrix: ", A.shape, "\n", A)
    print("maximum:", stat['max'])
    print("mean:", stat['mean'])
    print("standard deviation:", stat['stdev'])
    print("variance:", stat['var'])


# ## 3. Preprocessing Data
# 

# ### 3.1. Preprocessing Input Data
# 
# We will write the data processing functions in the section. There are three steps for input data:
# 
# 1. Standardize the data
# 2. Expand the features by window slice
# 3. Undersampling Data
# 

# #### 3.1.1. Standardize the data
# 
# Next, we will standardize the data by calculating **stat** from the previous step. 
# 
# How to standardize the data: https://stackoverflow.com/a/4544459
# 

# In[ ]:


def standardize(A, stat):
    # if isinstance(A, list):
    #     return [ standardize(a, stat) for a in A ]

    A = np.subtract(A, stat['mean'])
    A = np.divide(A, stat['stdev'])
    A = A.astype(np.float32)

    return A


# In[ ]:


if __name__ == "__main__":
    A = np.arange(2*3*6).reshape((2,3,6))
    stat = get_feat_stat([A])

    print("before standardization:", A.shape, "\n", A)
    A = standardize(A, stat)
    print("after standardization:", A.shape, "\n", A)


# #### 3.1.2. Expand the features by window slice
# 
# How to expand the features by window size: https://zhuanlan.zhihu.com/p/64933417
# 
# For example:
# 
# ```python
# A = np.arange(5*3*6).reshape((5,3,6))
# window_size = 5
# 
# A = np.arange(0,5*3*6,1).reshape((5,3,6)).astype(np.float128)
# A = np.pad(A, ((2,2), (2,2), (0,0)), mode='constant')
# A = strided(A, shape=(5,3,5,5,6), strides=(672,96,672,96,16))
# A = A.reshape((5,3,150))
# ```
# 
# **Args:**
# 
# - A: numpy array with shape (-1, -1, 6)
# - window_size: the size of window
# 
# **Returns:**
# 
# - A: numpy array with shape (-1, -1, window_size, window_size, 6)
# 

# In[ ]:


def expand(A, window_size):
    if not window_size & 0x1:
        raise Exception('need odd value on padding')

    # if isinstance(A, list):
    #     return [ expand(a, window_size) for a in A ]

    n = window_size # the height and width of the window
    p = window_size >> 1 # the padding size

    d0, d1, d2 = A.shape # dimansion 0, 1, 2
    s0, s1, s2 = A.strides # stride 0, 1, 2

    A = np.pad(A, pad_width=((p,p),(p,p),(0,0)), mode='constant')
    A = np.lib.stride_tricks.as_strided(A, shape=(d0,d1,n,n,d2), strides=(s0,s1,s0,s1,s2))

    return A


# In[ ]:


if __name__ == "__main__":
    A = np.arange(5*3*6).reshape((5,3,6))
    window_size = 5

    print("window size:", window_size)
    print("before expand:", A.shape, A.strides)
    A = expand(A, window_size)
    print("after expand:", A.shape, A.strides)
    print(A[-1][-1][-1])


# #### 3.1.3. Undersampling Data
# 
# Because of the large data, we need to undersample the data.
# 
# **Args:**
# 
# - A: numpy array with two or more dimension.
# - index: the sample indices of double array
# 
# **Returns:**
# 
# - A: one dimension less than the input numpy array.

# In[ ]:


def undersample(A, index):
    # if isinstance(A, list):
    #     return [undersample(a, i) for a, i in zip(A, index)]
    return A[index[:,1],index[:,0]]


# In[ ]:


if __name__ == "__main__":
    A = np.arange(2*3*5*5*6).reshape((2,3,5,5,6))
    sample = np.array([[2,1],[0,1],[1,0]])

    print("the sample indices:\n", sample)
    print("before undersampling shape:", A.shape)
    A = undersample(A, sample)
    print("after undersampling shape:", A.shape)


# ### 3.2. Preprocessing Output Data
# 
# The target will be classified into two categories. 
# 
# convert target value
# - if zero => 0
# - else => 1
# 
# **Args:**
# 
# - A: numpy array with shape (-1, 1).
# 
# **Returns:**
# 
# - A: numpy array with shape (-1, 1).

# In[ ]:


def classify(A):
    if isinstance(A, list):
        return [ classify(a) for a in A ]
    
    # zero => 0
    # else => 1
    
    A = A != 0
    A = A.astype(int)

    return A


# In[ ]:


if __name__ == "__main__":
    A = np.arange(2*3*1).reshape((2,3,1))

    print("before classification:\n", A)
    A = classify(A)
    print("after classification:\n", A)


# ## 4. Piping Preprocessing Data

# ### 4.1 Single Pipeline
# 
# Currently, it's a little bit complicated to piping the data, so we just iterated all preprocessing data functions instead.
# 
# **Args:**
# 
# - X: numpy array with shape (-1, -1, 6)
# - Y: numpy array with shape (-1, 1)
# - statistic: dictionary with keys "mean" and "stdev" used to standardize the data
# - window_size: the size of window
# - sample: the sample indices of double array
# 
# **Returns:**
# 
# - X: numpy array with shape (-1, window_size * window_size * 6)
# - Y: numpy array with shape (-1, 1)

# In[ ]:


def pipeline(X, Y, statistic, window_size=1, sample=None):
    
    w = window_size
    dx = X.shape[2]
    dy = Y.shape[2]
    
    X = standardize(X, statistic)
    X = expand(X, window_size)
    
    Y = classify(Y)
    
    if sample is not None:
        X = undersample(X, sample)
        Y = undersample(Y, sample)
        
    X = X.reshape((-1, w*w*dx))
    Y = Y.reshape((-1, dy))
    
    return (X, Y)


# In[ ]:


if __name__ == "__main__":
    X = np.arange(2*3*6).reshape((2,3,6))
    Y = np.arange(2*3*1).reshape((2,3,1))
    stat = get_feat_stat([X])
    w = 5
    s = np.array([[0,1], [2,1]])

    print("before preprocessing:")
    print("the shape of X:", X.shape)
    print("the shape of Y:", Y.shape)
    print("statistic:", stat)
    print("window size:", w)
    print("the size of sample:", len(s))
    X, Y = pipeline(X, Y, stat, window_size=w, sample=s)
    print("after preprocessing:")
    print("the shape of X:", X.shape)
    print("the shape of Y:", Y.shape)


# ### 4.2 Iterative All Pipeline
# 
# The different between the 4.1 and 4.2 is the type of input data. The type of 4.1 input data is numpy array and the type of 4.2 input data is the list of numpy array.
# 
# **Args:**
# 
# - X: the list of numpy array. reference 4.1
# - Y: the list of numpy array. reference 4.1
# - statistic: dictionary with keys "mean" and "stdev" used to standardize the data
# - window_size: the size of window
# - sample: the list of sample data. reference 4.1
# - callback: the callback function you want to execute. It will give you the data X and Y after processing function 4.1.
# 
# **Returns:**
# 
# None

# In[ ]:


def iterative_all(X_lists, Y_lists, statistic, window_size=1, samples=[], callback=None):
    
    if len(X_lists) != len(Y_lists):
        raise Exception('the length of X lists ({}) and Y lists ({}) are not the same'.format(len(X_lists), len(Y_lists)))
    
    if callback is None:
        raise Exception('iterative_all callback is NoneType')

    iters = zip(X_lists, Y_lists, samples)
    if len(X_lists) > len(samples):
        iters = itertools.zip_longest(X_lists, Y_lists, samples)
    
    w = window_size

    for x, y, s in iters:
        
        # show progressing information.
        d0, d1, dx = x.shape # dim
        s0 = d0*d1 if s is None else len(s) # sample len
        print("data size:", d0*d1, x.shape, "-->", (s0, w*w*dx), end=' ')
        
        # run.
        x, y = pipeline(x, y, statistic, window_size=w, sample=s)
    
        print(utility.sizeof_fmt(x.nbytes), utility.sizeof_fmt(y.nbytes))

        callback(x, y)


# In[ ]:


if __name__ == "__main__":
    X1 = np.arange(2*3*6).reshape((2,3,6))
    Y1 = np.arange(2*3*1).reshape((2,3,1))
    
    X2 = np.arange(3*4*6).reshape((3,4,6))
    Y2 = np.arange(3*4*1).reshape((3,4,1))
    
    X_lists = [X1, X2]
    Y_lists = [Y1, Y2]
    stat = get_feat_stat([X1, X2])
    w = 5
    
    s = [np.array([[0,1], [2,1]]), np.array([[1,2], [3,2]])]

    def _callback(X, Y):
        print("callback X shape:", X.shape)
        print("callback Y shape:", Y.shape)
    
    print("iterative all preprocessing:")
    print("the shape of X:", X1.shape, X2.shape)
    print("the shape of Y:", Y1.shape, Y2.shape)
    print("statistic:", stat)
    print("window size:", w)
    print("the size of sample:", len(np.concatenate(s, axis=0)))
    iterative_all(X_lists, Y_lists, stat, window_size=w, samples=s, callback=_callback)


# In[ ]:




