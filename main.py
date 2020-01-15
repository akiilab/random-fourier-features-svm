#!/usr/bin/env python
# coding: utf-8

# # SVM Kernel Approximation
# 
# 1. Environment Setup: import required libraries
# 2. Loading Dataset: training set, validation set, test set
# 3. Preprocessing Data
# 4. Undersampling Data
# 5. Concatenating Data
# 6. Building the Model
# 7. Training Data
# 8. Evalutaing Data
# 

# ## 1. Environment Setup: import required library
# 
# We include the required libraries that will be used in the next parts. The **time**, **numpy**, and **tensorflow** are common libraries in machine learning. The **fio**, which is "file input output" used to load data and **config**, which is "configuration file" used to config the path of the dataset files are written by myself. Modify it when you need it.

# In[ ]:


import fio
from config import *

import time
import numpy as np
import tensorflow as tf


# ## 2. Loading Dataset: training set, validation set, test set
# 
# Loading the training set, validation set that was defined in **config.py** file. The sample files consisting of indices of data will be used to undersample the data.
# 

# In[ ]:


X_train_orig = fio.load_file(train_data_dict['fcsv_phs'])
Y_train_orig = fio.load_file(train_data_dict['tcsv_phs'])
X_valid_orig = fio.load_file(eval_data_dict['fcsv_phs'])
Y_valid_orig = fio.load_file(eval_data_dict['tcsv_phs'])
train_sample = fio.load_sample_file(train_dataset_dict['Short-TrainSet-UdrSamp-3_3_1p0_1p0_0p1'])
valid_sample = fio.load_sample_file(valid_dataset_dict['Short-ValidSet-NoUdrSamp'])

print("the length of training set:", len(X_train_orig))
print("the length of evaluating set:", len(X_valid_orig))
print("the first row of training data:", X_train_orig[0][0][0])
print("the target value:", Y_train_orig[0][0][0])


# ## 3. Preprocessing Data
# 
# There are two parts of the data that must be processed:
# 
# - the input data represented by the prefix "**X**" on variables
# - the target data represented by the prefix "**Y**" on variables
# 
# The data will be processed on the following steps.
# 

# ### 3.1. Preprocessing Input Data
# 
# Dealing with the input data has three steps:
# 
# 1. Calculating the feature statistic data
# 2. Standardize the data
# 3. Expand the features by window slice
# 

# #### 3.1.1. Calculating the feature statistic data
# 
# we will write a function to calculate the training and validating data. The mean, maximum, standard deviation, and variance are returned at the end of the function.
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


stat = get_feat_stat(X_train_orig + X_valid_orig)
print("maximum:", stat['max'])
print("mean:", stat['mean'])
print("standard deviation:", stat['stdev'])
print("variance:", stat['var'])


# #### 3.1.2. Standardize the data
# 
# Next, we will standardize the data by the **stat** calculated from the previous step. 
# 
# How to standardize the data: https://stackoverflow.com/a/4544459
# 

# In[ ]:


def standardize(A, stat):
    if isinstance(A, list):
        return [ standardize(a, stat) for a in A ]

    A = np.subtract(A, stat['mean'])
    A = np.divide(A, stat['stdev'])
    A = A.astype(np.float32)

    return A


# In[ ]:


print("before standardization:", X_train_orig[0][0][0])
X_train_orig = standardize(X_train_orig, stat)
X_valid_orig = standardize(X_valid_orig, stat)
print("after standardization:", X_train_orig[0][0][0])


# #### 3.1.3. Expand the features by window slice
# 
# How to expand the features by window size: https://zhuanlan.zhihu.com/p/64933417
# 

# In[ ]:


def expand(A, window_size):
    if not window_size & 0x1:
        raise Exception('need odd value on padding')

    if isinstance(A, list):
        return [ expand(a, window_size) for a in A ]

    # For example
    # A is a (5,3,6) matrix
    # window_size is 5
    # 
    # A = np.arange(0,5*3*6,1).reshape((5,3,6)).astype(np.float128)
    # A = np.pad(A, ((2,2), (2,2), (0,0)), mode='constant')
    # A = strided(A, shape=(5,3,5,5,6), strides=(672,96,672,96,16))
    # A = A.reshape((5,3,150))
    #
    # For more information:
    # https://zhuanlan.zhihu.com/p/64933417

    n = window_size # the height and width of the window
    p = window_size >> 1 # the padding size

    d0, d1, d2 = A.shape # dimansion 0, 1, 2
    s0, s1, s2 = A.strides # stride 0, 1, 2

    A = np.pad(A, pad_width=((p,p),(p,p),(0,0)), mode='constant')
    A = np.lib.stride_tricks.as_strided(A, shape=(d0,d1,n,n,d2), strides=(s0,s1,s0,s1,s2))
    print("reshape: (", d0,d1,n,n,d2, ") -> (", d0,d1,d2*n*n, ")")
    A = A.reshape((d0,d1,d2*n*n))

    return A


# In[ ]:


window_size = 5
X_train = expand(X_train_orig, window_size)
X_valid = expand(X_valid_orig, window_size)
print("before expand:", X_train_orig[0].shape)
print("after expand:", X_train[0].shape)


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


Y_train = classify(Y_train_orig)
Y_valid = classify(Y_valid_orig)
print("before classification:", Y_train_orig[0][0][0])
print("after classification:", Y_train[0][0][0])


# ## 4. Undersampling Data
# 
# Because of the large data, we need to undersample the data.
# 

# In[ ]:


def undersample(arr, idx):
    if isinstance(arr, list):
        return [undersample(f, i) for f, i in zip(arr, idx)]
    return arr[idx[:,1],idx[:,0]]


# In[ ]:


print("the size of training sample:", len(train_sample[0]))
print("the first element of training sample:", train_sample[0][0])
print("before undersampling shape:", X_train[0].shape)
X_train = undersample(X_train, train_sample)
X_valid = undersample(X_valid, valid_sample)
Y_train = undersample(Y_train, train_sample)
Y_valid = undersample(Y_valid, valid_sample)
print("after undersampling shape:", X_train[0].shape)


# ## 5. Concatenating Data
# 
# The type of list cannot be trained by tensorflow, so we need to convert data from a **list** to a **numpy array**.

# In[ ]:


print("the type before concatenation:", type(X_train))
X_train = np.concatenate(X_train)
X_valid = np.concatenate(X_valid)
Y_train = np.concatenate(Y_train)
Y_valid = np.concatenate(Y_valid)
print("the type after concatenation:", type(X_train))


# ## 6. Building the Model
# 
# The model that we used is followed by the article: [Improving Linear Models Using Explicit Kernel Methods](https://github.com/Debian/tensorflow/blob/master/tensorflow/contrib/kernel_methods/g3doc/tutorial.md).

# In[ ]:


learining_rate = 50.0
l2_regularization_strength = 0.001

# Random Fourier Feature Mapper
dim_in  = window_size * window_size * 6
dim_out = window_size * window_size * 6 * 10
stddev  = 5.0

optimizer = tf.train.FtrlOptimizer(learning_rate=learining_rate, l2_regularization_strength=l2_regularization_strength)

image_column = tf.contrib.layers.real_valued_column('data', dimension=dim_in)
kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(input_dim=dim_in, output_dim=dim_out, stddev=stddev, name='rffm')

estimator = tf.contrib.kernel_methods.KernelLinearClassifier(n_classes=2, optimizer=optimizer, kernel_mappers={image_column: [kernel_mapper]})

# For Example: Linear Model without optimizer
# image_column = tf.contrib.layers.real_valued_column('data', dimension=784)
# estimator = tf.contrib.learn.LinearClassifier(feature_columns=[image_column], n_classes=2)


# ## 7. Training Data
# 

# In[ ]:


batch = 2
epoch = 1
steps = 2000

x = {'data':X_train}
y = Y_train
train_input_fn = tf.estimator.inputs.numpy_input_fn(x, y, batch_size=batch, shuffle=False, num_epochs=epoch)

# Train.
start = time.time()
estimator.fit(input_fn=train_input_fn, steps=2000)
end = time.time()
print('Elapsed time: {} seconds'.format(end - start))


# ## 8. Evalutaing Data

# In[ ]:


x = {'data':X_valid}
y = Y_valid

eval_input_fn = tf.estimator.inputs.numpy_input_fn(x, y, batch_size=2, shuffle=False, num_epochs=1)

# Evaluate and report metrics.
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
print(eval_metrics)


# In[ ]:




