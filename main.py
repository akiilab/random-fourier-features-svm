import fio
import common
from config import *

import numpy as np
import tensorflow as tf

X_train_orig = fio.load_file(train_data_dict['fcsv_phs'])
Y_train_orig = fio.load_file(train_data_dict['tcsv_phs'])
X_valid_orig = fio.load_file(eval_data_dict['fcsv_phs'])
Y_valid_orig = fio.load_file(eval_data_dict['tcsv_phs'])
train_sample = fio.load_sample_file(train_dataset_dict['Short-TrainSet-UdrSamp-3_3_1p0_1p0_0p1'])
valid_sample = fio.load_sample_file(valid_dataset_dict['Short-ValidSet-NoUdrSamp'])


stat = common.get_feat_stat(X_train_orig + X_valid_orig)
X_train_orig = common.standardize(X_train_orig, stat)
X_valid_orig = common.standardize(X_valid_orig, stat)

window_size = 5
X_train = common.expand(X_train_orig, window_size)
X_valid = common.expand(X_valid_orig, window_size)

Y_train = common.classify(Y_train_orig)
Y_valid = common.classify(Y_valid_orig)


X_train = common.undersample(X_train, train_sample)
X_valid = common.undersample(X_valid, valid_sample)
Y_train = common.undersample(Y_train, train_sample)
Y_valid = common.undersample(Y_valid, valid_sample)

X_train = np.concatenate(X_train)
X_valid = np.concatenate(X_valid)
Y_train = np.concatenate(Y_train)
Y_valid = np.concatenate(Y_valid)

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

x = {'data':X_train}
y = Y_train

train_input_fn = tf.estimator.inputs.numpy_input_fn(x, y, batch_size=2, shuffle=False, num_epochs=1)

x = {'data':X_valid}
y = Y_valid

eval_input_fn = tf.estimator.inputs.numpy_input_fn(x, y, batch_size=2, shuffle=False, num_epochs=1)

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

