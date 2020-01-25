#!/usr/bin/env python
# coding: utf-8

# # SVM Kernel Approximation
# 
# 1. Environment Setup: import required libraries
# 2. Loading Dataset: training set, validation set, test set
# 3. Preprocessing Data
# 4. Building the Model
# 5. Training Data
# 6. Validating Data
# 7. Testing Data
# 

# ## 1. Environment Setup: import required library
# 
# We include the required libraries that will be used in the next parts. The **time**, **numpy**, and **tensorflow** are common libraries in machine learning. The **fio**, which is "file input output" used to load data and **config**, which is "configuration file" used to config the path of the dataset files are written by myself. Modify it when you need it.

# In[ ]:


import fio
import preprocessing as pc
from config import *
import utility

# python std library
import gc
import time

# install library
import numpy as np
import tensorflow as tf


# ## 2. Loading Dataset: training set, validation set, test set
# 
# Loading the training set, validation set, and test set that was defined in **config.py** file. Sample files consisting of indices of data will be used to undersample the data.
# 

# In[ ]:


X_train_orig = fio.load_file(X_train_dataset)
Y_train_orig = fio.load_file(Y_train_dataset)
X_test_orig = fio.load_file(X_test_dataset)
Y_test_orig = fio.load_file(Y_test_dataset)

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
# Check out the file **preprocess.py** for more details.
# 

# In[ ]:


def preprocessing(X_lists, Y_lists, statistic, window_size=1, samples=[]):
    
    X = []
    Y = []
    
    def _callback(x, y):
        X.append(x)
        Y.append(y)
        
    pc.iterative_all(X_lists, Y_lists, statistic, window_size=window_size, samples=samples, callback=_callback)
        
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    
    X = X.reshape((X.shape[0],-1))
    Y = Y.reshape((Y.shape[0],-1))
    
    print(X.shape, utility.sizeof_fmt(X.nbytes))
    print(Y.shape, utility.sizeof_fmt(Y.nbytes))
    
    return (X, Y)
    


# In[ ]:


w = 23 # window size
stat = pc.get_feat_stat(X_train_orig)

train_sample = fio.load_sample_file(train_sample_dataset)
X_train, Y_train = preprocessing(X_train_orig, Y_train_orig, stat, window_size=w, samples=train_sample)


# ## 4. Building the Model
# 
# The model that we used is followed by the article: [Improving Linear Models Using Explicit Kernel Methods](https://github.com/Debian/tensorflow/blob/master/tensorflow/contrib/kernel_methods/g3doc/tutorial.md).
# 
# https://storage.googleapis.com/pub-tools-public-publication-data/pdf/18d86099a350df93f2bd88587c0ec6d118cc98e7.pdf

# In[ ]:


def create_model(dim_in, dim_out, stddev=5.0, learning_rate=0.003, l2_regularization_strength=0.006):
    
    optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l2_regularization_strength=l2_regularization_strength)
    
    image_column = tf.contrib.layers.real_valued_column('data', dimension=dim_in)
    kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(input_dim=dim_in, output_dim=dim_out, stddev=stddev, name='rffm')

    estimator = tf.contrib.kernel_methods.KernelLinearClassifier(n_classes=2, optimizer=optimizer, kernel_mappers={image_column: [kernel_mapper]})

    return estimator
    
    # For Example: Linear Model without optimizer
    # image_column = tf.contrib.layers.real_valued_column('data', dimension=784)
    # estimator = tf.contrib.learn.LinearClassifier(feature_columns=[image_column], n_classes=2)


# In[ ]:


dim_in  = w * w * 6
dim_out = w * w * 6 * 10
stddev  = 5.0
learning_rate = 0.003
l2_regularization_strength = 0.006

estimator = create_model(dim_in, dim_out, stddev, learning_rate, l2_regularization_strength)


# ## 5. Training Data
# 

# In[ ]:


def train_model(estimator, X, Y, batch_size=128, epoch=1):
    print("training data shape:", X.shape, Y.shape)
    print("training data memory:", utility.sizeof_fmt(X.nbytes), utility.sizeof_fmt(Y.nbytes))

    x = {'data':X}
    y = Y
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x, y, batch_size=batch, shuffle=True, num_epochs=epoch)
    
    start = time.time()
    estimator.fit(input_fn=train_input_fn) # Train.
    end = time.time()
    print('Elapsed time: {} seconds'.format(end - start))
    


# In[ ]:


batch = 128
epoch = 1

train_model(estimator, X_train, Y_train, batch, epoch)


# ## 6. Validating Data
# 
# 1. Evaluating Training Data
# 2. Evaluating Validation Data
# 3. Get The Prediction Data
# 
# metrics
# 
# https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/metrics
# 
# 
# true false positive negative
# 
# https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative
# 
# 
# https://ai.stackexchange.com/questions/6383/meaning-of-evaluation-metrics-in-tensorflow
# 
# - loss: The current value of the loss. Either the sum of the losses, or the loss of the last batch.
# - global_step: Number of iterations.
# - AUC or Area Under the (ROC) Curve is quite complicated, but tells you something about the true/false positive rate. In short: the AUC is equal to the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one.
# - auc_precision_recall: Is the percentage of relevant intstances, among the retrieved instances, that have been retrieved over the total amount of relevant instances.
# 

# In[ ]:


def evaluate_model(estimator, X, Y, statistic, batch=None):
    
    if batch == None:
        batch = X.shape[0]
    
    x = {'data': X}
    y = Y
    
    input_fn = tf.estimator.inputs.numpy_input_fn(x, y, batch_size=batch, shuffle=False, num_epochs=1)
    
    metric = estimator.evaluate(input_fn=input_fn)
    metric['count'] = X.shape[0]
    
    return metric

def evaluate_models(estimator, X_lists, Y_lists, statistic, window_size=1, samples=[], batch=None):

    metrics = []
    def _callback(X, Y):
        metric = evaluate_model(estimator, X, Y, statistic, batch=batch)
        metrics.append(metric)
        print("evaluated metrics:", metric)

    pc.iterative_all(X_lists, Y_lists, statistic, window_size=window_size, samples=samples, callback=_callback)

    return metrics


# ### 6.1 Evaluating Training Data

# In[ ]:


metrics = evaluate_model(estimator, X_train, Y_train, stat, batch=128)
print("training metrics (grouped):", metrics)
metrics = evaluate_models(estimator, X_train_orig, Y_train_orig, stat, window_size=w, samples=train_sample, batch=128)
print("training metrics:", metrics)

# metric = evaluate_model(estimator, X_train, Y_train, stat, batch=128)
# del X_train, Y_train
# gc.collect()


# ### 6.2 Evaluating Validation Data

# In[ ]:


valid_sample = fio.load_sample_file(valid_sample_dataset)
metrics = evaluate_models(estimator, X_train_orig, Y_train_orig, stat, window_size=w, samples=valid_sample, batch=128)
print("validation metrics:", metrics)


# In[ ]:


# X_valid, Y_valid = preprocessing(X_train_orig, Y_train_orig, stat, window_size=w, samples=valid_sample)
# metric = evaluate_model(estimator, X_valid, Y_valid, stat, batch=128)
# del X_valid, Y_valid
# gc.collect()


# ### 6.3 Get The Prediction Data
# 
# ```python
# x = {'data': X_train.astype(np.float32)}
# y = Y_train
# 
# input_fn = tf.estimator.inputs.numpy_input_fn(x, batch_size=batch, shuffle=False)
# metric = estimator.predict(input_fn=input_fn)
# print(metric['classes'])
# ```
# 
# **Args:**
# 
# - input_fn: Input function. If set, `x` and 'batch_size' must be `None`.
# 
# **Returns:**
# 
# A numpy array of predicted classes or regression values if the constructor's `model_fn` returns a `Tensor` for `predictions` or a `dict` of numpy arrays if `model_fn` returns a `dict`.

# ## 7. Testing Data

# In[ ]:


metrics = evaluate_models(estimator, X_test_orig, Y_test_orig, stat, window_size=w, batch=128)
print("testing metrics:", metrics)


# In[ ]:




