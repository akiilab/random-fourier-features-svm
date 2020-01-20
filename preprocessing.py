#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing Library
# 
# 1. Environment Setup: import required libraries
# 2. Calculating Features Statistic Data
# 3. Preprocessing Data
# 4. Building the Model
# 5. Training Data
# 6. Evalutaing Data
# 

# ## 1. Environment Setup: import required library
# 

# In[ ]:


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
#     if isinstance(A, list):
#         return [ standardize(a, stat) for a in A ]

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

# In[ ]:


def expand(A, window_size):
    if not window_size & 0x1:
        raise Exception('need odd value on padding')

#     if isinstance(A, list):
#         return [ expand(a, window_size) for a in A ]

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

# In[ ]:


def undersample(arr, idx):
    if isinstance(arr, list):
        return [undersample(f, i) for f, i in zip(arr, idx)]
    return arr[idx[:,1],idx[:,0]]


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
# - the target value is zero 
# - the target vlaue is not zero

# In[ ]:


def classify(A):
    if isinstance(A, list):
        return [ classify(a) for a in A ]

    A = A != 0
    A = A.astype(int)

    return A


# In[ ]:


if __name__ == "__main__":
    A = np.arange(2*3*1).reshape((2,3,1))

    print("before classification:\n", A)
    A = classify(A)
    print("after classification:\n", A)


# In[ ]:




