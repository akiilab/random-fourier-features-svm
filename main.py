import fio
import common
from config import *

import numpy as np

X_train_orig = fio.load_file(train_data_dict['fcsv_phs'])
Y_train_orig = fio.load_file(train_data_dict['tcsv_phs'])
X_valid_orig = fio.load_file(eval_data_dict['fcsv_phs'])
Y_valid_orig = fio.load_file(eval_data_dict['tcsv_phs'])
train_sample = fio.load_sample_file(train_dataset_dict['Short-TrainSet-UdrSamp-3_3_1p0_1p0_0p1'])
valid_sample = fio.load_sample_file(valid_dataset_dict['Short-TrainSet-NoUdrSamp'])


stat = common.get_feat_stat(X_train_orig + X_valid_orig)
X_train_orig = common.standardize(X_train_orig, stat)
X_valid_orig = common.standardize(X_valid_orig, stat)
X_train = common.expand(X_train_orig, 5)
X_valid = common.expand(X_valid_orig, 5)

Y_train = common.classify(Y_train_orig)
Y_valid = common.classify(Y_valid_orig)


X_train = common.undersample(X_train, train_sample)
X_valid = common.undersample(X_valid, valid_sample)
Y_train = common.undersample(Y_train, train_sample)
Y_valid = common.undersample(Y_valid, valid_sample)

X_train = np.concatenate(X_train).astype(np.float32)
X_valid = np.concatenate(X_valid).astype(np.float32)
Y_train = np.concatenate(Y_train)
Y_valid = np.concatenate(Y_valid)

import tensorflow as tf
x = {'data':X_train}
y = Y_train

train_input_fn = tf.estimator.inputs.numpy_input_fn(x, y, batch_size=2, shuffle=False, num_epochs=1)

x = {'data':X_valid}
y = Y_valid

eval_input_fn = tf.estimator.inputs.numpy_input_fn(x, y, batch_size=2, shuffle=False, num_epochs=1)

image_column = tf.contrib.layers.real_valued_column('data', dimension=5*5*6)
estimator = tf.contrib.learn.LinearClassifier(feature_columns=[image_column], n_classes=2)

# Train.
import time
start = time.time()
estimator.fit(input_fn=train_input_fn, steps=2000)
end = time.time()
print('Elapsed time: {} seconds'.format(end - start))

# Evaluate and report metrics.
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
print(eval_metrics)

import ipdb; ipdb.set_trace()

