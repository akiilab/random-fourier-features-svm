#!/usr/bin/env python
# coding: utf-8

# # SVM Kernel Approximation
# 
# 1. Environment Setup: import required libraries
# 2. Loading Dataset and Preprocessing Data
# 3. Build Input Fn
# 4. Training Model
# 5. Evaluating Training Data
# 

# ## 1. Environment Setup: import required library
# 
# We include the required libraries that will be used in the next parts. The **time**, **numpy**, and **tensorflow** are common libraries in machine learning. The **fio**, which is "file input output" used to load data and **config**, which is "configuration file" used to config the path of the dataset files are written by myself. Modify it when you need it.
# 
# **patch/metrics.py**: fix `tf.metrics.true_negatives` method is missing on Tensorflow r1.4.

# In[ ]:


import fio
import preprocessing as pc
from config import *
import utility

# python std library
import gc
import time
import logging
import functools
import collections
from multiprocessing.pool import ThreadPool

# install library
import numpy as np
import tensorflow as tf

# patch
import patch.metrics


# ## 2. Loading Dataset and Preprocessing Data
# 
# Loading the training set, validation set, and test set that was defined in **config.py** file. Sample files consisting of indices of data will be used to undersample the data.
# 
# There are two parts of the data that must be processed:
# 
# - the input data represented by the prefix "**X**" on variables
# - the target data represented by the prefix "**Y**" on variables
# 
# Check out the file **preprocess.py** for more details.

# In[ ]:


def load(window_size):
    X_train = fio.load_file(X_train_dataset)
    Y_train = fio.load_file(Y_train_dataset)
    X_test = fio.load_file(X_test_dataset)
    Y_test = fio.load_file(Y_test_dataset)
    train_sample = fio.load_sample_file(train_sample_dataset)
    valid_sample = fio.load_sample_file(valid_sample_dataset)

    stat = pc.get_feat_stat(X_train)
    
    X_train = pc.standardize(X_train, stat)
    X_test = pc.standardize(X_test, stat)
    
    X_train = pc.expand(X_train, window_size)
    X_test = pc.expand(X_test, window_size)
    
    Y_train = pc.classify(Y_train)
    Y_test = pc.classify(Y_test)
    
    return X_train, Y_train, X_test, Y_test, train_sample, valid_sample


# In[ ]:


if __name__ == "__main__":
    window_size = 23
    
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, train_sample, valid_sample = load(window_size)
    print("the length of training set:", len(X_train_orig))
    print("the length of testing set:", len(X_test_orig))
    print("the first row training data:", np.sum(X_train_orig[0][0][0]))
    print("the first row target value:", np.sum(Y_train_orig[0][0][0]))
    
    print(X_train_orig[0].shape)
    print(Y_train_orig[0].shape)
    print(train_sample[0])
    print(valid_sample[0])


# ## 3. Build Input Fn
# 
# - [Tensorflow Doc: dataset.from_generator](https://github.com/tensorflow/docs/blob/r1.4/site/en/api_docs/api_docs/python/tf/data/Dataset.md#from_generator)
# - [Tensorflow Doc: dataset.batch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch)
#     - batch_size: A tf.int64 scalar tf.Tensor, representing the number of consecutive elements of this dataset to combine in a single batch.
#     - drop_remainder: (Optional.) A tf.bool scalar tf.Tensor, representing whether the last batch should be dropped in the case it has fewer than batch_size elements; the default behavior is not to drop the smaller batch.
# - [Tensorflow Doc: dataset.padded_batch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#padded_batch)
# - [How to use dataset in tensorflow](https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428)
# - [StackOverflow: Train Tensorflow model with estimator (from_generator)](https://stackoverflow.com/questions/49673602/train-tensorflow-model-with-estimator-from-generator?rq=1)
# - [StackOverflow: Is Tensorflow Dataset API slower than Queues?](https://stackoverflow.com/questions/47403407/is-tensorflow-dataset-api-slower-than-queues)
# - [Github: How can I ues Dataset to shuffle a large whole dataset?](https://github.com/tensorflow/tensorflow/issues/14857)
# 
# **Got the warning: Out of range StopIteration**
# 
# ```shell
# W tensorflow/core/framework/op_kernel.cc:1192] Out of range: StopIteration: Iteration finished
# ```
# 
# > I also meeting this problem same for you,but it is not a bug.
# >
# > you can see the doc in https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator about train()
# > 
# > steps: Number of steps for which to train model. If None, train forever or train until input_fn generates the OutOfRange error or StopIteration exception. 'steps' works incrementally. If you call two times train(steps=10) then training occurs in total 20 steps. If OutOfRange or StopIteration occurs in the middle, training stops before 20 steps. If you don't want to have incremental behavior please set max_steps instead. If set, max_steps must be None.
# >
# > -- libulin
# 
# From [Github Comment](https://github.com/tensorflow/tensorflow/issues/12414#issuecomment-345131765)
# 
# With the fix in [301a6c4](https://github.com/tensorflow/tensorflow/commit/301a6c41cbb111fae89657a49775920aa70525fd) (and a related fix for the StopIteration logging in [c154d47](https://github.com/tensorflow/tensorflow/commit/c154d4719eea88e694f4c06bcb1249dbac0f7877), the logs should be much quieter when using tf.data.
# 
# Simple fix:
# 
# ```python
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
# import tensorflow as tf
# ```

# In[ ]:


def np_input_fn(X, Y, samples=[], shuffle=False, window_size=1, batch=None, epoch=None):
    if batch is None:
        raise Exception('batch can not be None')
    
    if window_size & 0x1 == 0:
        raise Exception('window size can not even')
    dim_in = window_size * window_size * 6
    
    if not isinstance(X, list):
        X = [X]
    if not isinstance(Y, list):
        Y = [Y]
    if not isinstance(samples, list):
        samples = [samples]
    
    if len(samples) == 0:
        samples = [np.indices((x.shape[0], x.shape[1])).reshape((2,-1)).T for x in X]
    
    samples = [np.pad(x, ((0,0),(1,0)), 'constant', constant_values=i) for i, x in enumerate(samples)]
    samples = np.concatenate(samples)
    
    print("input_fn total size", len(samples))
    
    def generator():
        if shuffle == True:
            np.random.shuffle(samples)
        
        for s in samples:
            x = X[s[0]][s[1], s[2]].reshape((dim_in))
            y = Y[s[0]][s[1], s[2]].reshape((1))
            yield x, y
    
    def _input_fn():
        dataset = tf.data.Dataset.from_generator(generator,
                                                   output_types= (tf.float32, tf.int32), 
                                                   output_shapes=(tf.TensorShape([dim_in]), tf.TensorShape([1])))
        dataset = dataset.batch(batch_size=batch)
        dataset = dataset.repeat(epoch)
        dataset = dataset.prefetch(1)

        iterator = dataset.make_one_shot_iterator()
        features_tensors, labels = iterator.get_next()
        print(features_tensors)
        print(labels)
        features = {'data': features_tensors }
        return features, labels
    
    return _input_fn


# In[ ]:


if __name__ == "__main__":
    print(np_input_fn(X_train_orig, Y_train_orig, train_sample, window_size=23, batch=128)())


# ## 4. Training Model
# 
# The model that we used is followed by the article: [Improving Linear Models Using Explicit Kernel Methods](https://github.com/Debian/tensorflow/blob/master/tensorflow/contrib/kernel_methods/g3doc/tutorial.md).
# 
# [TensorFlow Estimators: Managing Simplicity vs. Flexibility in
# High-Level Machine Learning Frameworks](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/18d86099a350df93f2bd88587c0ec6d118cc98e7.pdf)
# 
# Optimizer
# 
# - [Ftrl Optimizer](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/FtrlOptimizer)
# - [Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer)
# 
# Build the following models:
# 
# 1. Build Linear Classifier Model
# 2. Build Random Fourier Feature Mapper Model and Linear Classifier Model

# ### 4.1. Training Linear Classifier Model

# In[ ]:


def create_linear_model(learning_rate, dim_in, config=None):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    image_column = tf.contrib.layers.real_valued_column('data', dimension=dim_in)
    
    estimator = tf.contrib.learn.LinearClassifier(
        feature_columns=[image_column],
        n_classes=2, 
        config=config,
        optimizer=optimizer)

    return estimator


# In[ ]:


if __name__ == "__main__":
    batch = 128
    epoch = 2
    train_input_fn = np_input_fn(X_train_orig, Y_train_orig, train_sample, shuffle=True, window_size=23, batch=batch, epoch=epoch)
    
    learning_rate = 0.001       # Adam Optimizer
    input_dim = 23 * 23 * 6     # Data size
    
    estimator = create_linear_model(learning_rate, input_dim)

    start = time.time()
    estimator.fit(input_fn=train_input_fn) # Train.
    end = time.time()
    print('Elapsed time: {} seconds'.format(end - start))
    


# ### 4.2. Training Random Fourier Feature Mapper Model

# In[ ]:


def create_rffm_model(learning_rate, dim_in, dim_out, stddev, config=None):
    
    kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(dim_in, dim_out, stddev, name='rffm')
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    image_column = tf.contrib.layers.real_valued_column('data', dimension=dim_in)

    estimator = tf.contrib.kernel_methods.KernelLinearClassifier(
        feature_columns=[image_column], 
        n_classes=2, 
        config=config,
        optimizer=optimizer, 
        kernel_mappers={image_column: [kernel_mapper]})
    
    return estimator


# In[ ]:


if __name__ == "__main__":
    
    batch = 128
    epoch = 1
    train_input_fn = np_input_fn(X_train_orig, Y_train_orig, train_sample, shuffle=True, window_size=23, batch=batch, epoch=epoch)
    
    learning_rate = 0.001  # Adam Optimizer

    # RFFM
    input_dim = 23 * 23 * 6
    output_dim = 23 * 23 * 6 * 10
    stddev = 1.0

    estimator = create_rffm_model(learning_rate, input_dim, output_dim, stddev)
    
    start = time.time()
    estimator.fit(input_fn=train_input_fn) # Train.
    end = time.time()
    print('Elapsed time: {} seconds'.format(end - start))


# ## 5. Evaluating Training Data
# 
# 1. Evaluating Training Data
# 2. Evaluating Validation Data
# 3. Evaluating Testing Data
# 
# **Confusion Matrix**
# 
# - [Classification: True vs. False and Positive vs. Negative](https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative)
# - [如何辨別機器學習模型的好壞？秒懂Confusion Matrix](https://www.ycc.idv.tw/confusion-matrix.html)
# 
# **estimator.evaluate**
# 
# - [Tensorflow Doc: estimator evaluate metrics](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/metrics)
# - [Stack Overflow: Meaning of evaluation metrics in Tensorflow](https://ai.stackexchange.com/questions/6383/meaning-of-evaluation-metrics-in-tensorflow)
# 
# ```python
# x, y = {'data': X}, Y
# input_fn = tf.estimator.inputs.numpy_input_fn(x, y, batch_size=batch, shuffle=False, num_epochs=1)
# metric = estimator.evaluate(input_fn=input_fn)
# ```
# 
# **estimator.predict_classes**
# 
# ```python
# x, y = {'data': X_train.astype(np.float32) }, Y_train
# batch = 128
# 
# input_fn = tf.estimator.inputs.numpy_input_fn(x, batch_size=batch, shuffle=False, num_epochs=1)
# metric = estimator.predict_classes(input_fn=input_fn)
# 
# for i, p in enumerate(metric):
#     print(p, y[i][0])
# ```
# 
# **Metrics**
# 
# - [Python tensorflow.contrib.learn.MetricSpec() Examples](https://www.programcreek.com/python/example/96156/tensorflow.contrib.learn.MetricSpec)
# - [Tensorflow Doc: Available Metrics](https://github.com/tensorflow/docs/tree/r1.4/site/en/api_docs/api_docs/python/tf/metrics)
# 
# ```python
# metrics = { "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes") }
# metric = estimator.evaluate(input_fn=eval_input_fn, metrics=metrics)
# ```
# 

# In[ ]:


def evaluate_model(estimator, X, Y, samples=[], window_size=1, batch=2048, epoch=1):

    eval_input_fn = np_input_fn(X, Y, samples, shuffle=False, window_size=window_size, batch=batch, epoch=epoch)
    
    metrics = {
        "tp": tf.contrib.learn.MetricSpec(metric_fn=tf.metrics.true_positives, prediction_key="classes"),
        "tn": tf.contrib.learn.MetricSpec(metric_fn=patch.metrics.true_negatives, prediction_key="classes"),
        "fp": tf.contrib.learn.MetricSpec(metric_fn=tf.metrics.false_positives, prediction_key="classes"),
        "fn": tf.contrib.learn.MetricSpec(metric_fn=tf.metrics.false_negatives, prediction_key="classes"),
    }
    
    start = time.time()
    metric = estimator.evaluate(input_fn=eval_input_fn, metrics=metrics)
    end = time.time()
    print('Elapsed time: {} seconds'.format(end - start))
    
    return metric


# ### 5.1 Evaluating Training Data

# In[ ]:


if __name__ == "__main__":
    start = time.time()
    metrics = evaluate_model(estimator, X_train_orig, Y_train_orig, batch=1, samples=train_sample, window_size=23)
    end = time.time()
    print('Elapsed time: {} seconds'.format(end - start))
    print("training metrics:", metrics)


# ### 5.2 Evaluating Validation Data

# In[ ]:


if __name__ == "__main__":
    start = time.time()
    metrics = evaluate_model(estimator, X_train_orig, Y_train_orig, batch=1, samples=valid_sample, window_size=23)
    end = time.time()
    print('Elapsed time: {} seconds'.format(end - start))
    print("validation metrics:", metrics)


# ### 5.3 Evaluating Testing Data

# In[ ]:


if __name__ == "__main__":
    print(len(X_test_orig))
    print(X_test_orig[0].shape)
    print(X_test_orig[1].shape)
    start = time.time()
    metrics = evaluate_model(estimator, X_test_orig, Y_test_orig, batch=1, window_size=23)
    end = time.time()
    print('Elapsed time: {} seconds'.format(end - start))
    print("testing metrics:", metrics)


# In[ ]:




