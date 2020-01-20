#!/usr/bin/env python
# coding: utf-8

# # SVM Kernel Approximation
# 
# 1. Environment Setup: import required libraries
# 2. Loading Dataset: training set, validation set, test set
# 3. Preprocessing Data
# 4. Building the Model
# 5. Training Data
# 6. Evalutaing Data
# 7. Testing Data
# 

# ## 1. Environment Setup: import required library
# 
# We include the required libraries that will be used in the next parts. The **time**, **numpy**, and **tensorflow** are common libraries in machine learning. The **fio**, which is "file input output" used to load data and **config**, which is "configuration file" used to config the path of the dataset files are written by myself. Modify it when you need it.

# In[ ]:


import fio

import preprocessing as pc
from config import *

import time
import itertools
import numpy as np
import tensorflow as tf


# ## 2. Loading Dataset: training set, validation set, test set
# 
# Loading the training set, validation set, and test set that was defined in **config.py** file. The sample files consisting of indices of data will be used to undersample the data.
# 

# In[ ]:


X_train_orig = fio.load_file(train_data_dict['fcsv_phs'])
Y_train_orig = fio.load_file(train_data_dict['tcsv_phs'])
X_test_orig = fio.load_file(eval_data_dict['fcsv_phs'])
Y_test_orig = fio.load_file(eval_data_dict['tcsv_phs'])

print("the length of training set:", len(X_train_orig))
print("the length of testing set:", len(X_test_orig))
print("the first row training data:", X_train_orig[0][0][0])
print("the first row target value:", Y_train_orig[0][0][0])


# ## 3. Preprocessing Data
# 
# There are two parts of the data that must be processed:
# 
# - the input data represented by the prefix "**X**" on variables
# - the target data represented by the prefix "**Y**" on variables
# 
# The data will be processed on the following steps.
# 

# In[ ]:


def pipeline(X, Y, statistic, window_size=1, sample=None):
    
    X = pc.standardize(X, statistic)
    X = pc.expand(X, window_size)
    
    Y = pc.classify(Y)
    
    if sample is not None:
        X = pc.undersample(X, sample)
        Y = pc.undersample(Y, sample)
    
    return (X, Y)


# In[ ]:


def preprocessing(X_lists, Y_lists, statistic, window_size=1, samples=[]):

    X = []
    Y = []
    
    for x, y, sample in itertools.zip_longest(X_lists, Y_lists, samples):
        x, y = pipeline(x, y, statistic, window_size, sample)
        X.append(x)
        Y.append(y)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    
    X = X.reshape((X.shape[0],-1))
    Y = Y.reshape((Y.shape[0],-1))
    
    return (X, Y)
    


# In[ ]:


w = 23 # window size
stat = pc.get_feat_stat(X_train_orig)
train_sample = fio.load_sample_file(train_dataset_dict['Short-TrainSet-UdrSamp-3_3_1p0_1p0_0p1'])
valid_sample = fio.load_sample_file(valid_dataset_dict['Short-ValidSet-NoUdrSamp'])

X_train, Y_train = preprocessing(X_train_orig, Y_train_orig, stat, window_size=w, samples=train_sample)
X_valid, Y_valid = preprocessing(X_train_orig, Y_train_orig, stat, window_size=w, samples=valid_sample)


# ## 4. Building the Model
# 
# The model that we used is followed by the article: [Improving Linear Models Using Explicit Kernel Methods](https://github.com/Debian/tensorflow/blob/master/tensorflow/contrib/kernel_methods/g3doc/tutorial.md).

# In[ ]:


learining_rate = 50.0
l2_regularization_strength = 0.001

# Random Fourier Feature Mapper
dim_in  = w * w * 6
dim_out = w * w * 6 * 10
stddev  = 5.0

optimizer = tf.train.FtrlOptimizer(learning_rate=learining_rate, l2_regularization_strength=l2_regularization_strength)

image_column = tf.contrib.layers.real_valued_column('data', dimension=dim_in)
kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(input_dim=dim_in, output_dim=dim_out, stddev=stddev, name='rffm')

estimator = tf.contrib.kernel_methods.KernelLinearClassifier(n_classes=2, optimizer=optimizer, kernel_mappers={image_column: [kernel_mapper]})

# For Example: Linear Model without optimizer
# image_column = tf.contrib.layers.real_valued_column('data', dimension=784)
# estimator = tf.contrib.learn.LinearClassifier(feature_columns=[image_column], n_classes=2)


# ## 5. Training Data
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
estimator.fit(input_fn=train_input_fn, steps=steps)
end = time.time()
print('Elapsed time: {} seconds'.format(end - start))

eval_metrics = estimator.evaluate(input_fn=train_input_fn, steps=1)
print("train data evaluated matrics:", eval_metrics)


# In[ ]:





# ## 6. Evalutaing Data

# In[ ]:


x = {'data':X_valid}
y = Y_valid

eval_input_fn = tf.estimator.inputs.numpy_input_fn(x, y, batch_size=2, shuffle=False, num_epochs=1)

# Evaluate and report metrics.
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
print("validation data evaluated matrics:", eval_metrics)


# ## 7. Testing Data
