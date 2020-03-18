#!/usr/bin/env python
# coding: utf-8

# ## Import Library

# In[ ]:


import tensorflow as tf
import svm
import time


# ## Experimental Functions
# 
# 1. Hooks
#     - get metrics at step n
#     - CheckpointSaverHook
#     - CheckpointHook
# 2. RunConfig
# 
# **Hooks**
# 
# - [An Advanced Example of Tensorflow Estimators Part (3/3)](https://medium.com/@tijmenlv/an-advanced-example-of-tensorflow-estimators-part-3-3-8c2efe8ff6fa)
# - [Tensorflow Github: tf.train.SessionRunContext](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/training/session_run_hook.py#L216)
# - [Tensorflow Doc: tf.train.SessionRunContext](https://github.com/tensorflow/docs/blob/r1.4/site/en/api_docs/api_docs/python/tf/train/SessionRunContext.md)
# - [Tensorflow Github: tf.train.SecondOrStepTimer](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/training/basic_session_run_hooks.py#L88)
# 
# **get metrics at step n**
# 
# ```python
# metrics = {
#     "tp": tf.contrib.learn.MetricSpec(metric_fn=tf.metrics.true_positives, prediction_key="classes"),
#     "tn": tf.contrib.learn.MetricSpec(metric_fn=patch.metrics.true_negatives, prediction_key="classes"),
#     "fp": tf.contrib.learn.MetricSpec(metric_fn=tf.metrics.false_positives, prediction_key="classes"),
#     "fn": tf.contrib.learn.MetricSpec(metric_fn=tf.metrics.false_negatives, prediction_key="classes"),
# }
# 
# 
# monitors = [
#    tf.contrib.learn.monitors.ValidationMonitor(input_fn=train_input_fn, every_n_steps=10, metrics=metrics)
# ]
#     
# estimator.fit(input_fn=train_input_fn, monitors=monitors)
# ```
# 
# **CheckpointSaverHook**: save checkpoint to *checkpoint_dir* every n steps
# **CheckpointHook**: save checkpoint to *checkpoint_dir* every n steps after save checkpoint every m steps
# 
# ```python
# class CheckpointHook(tf.train.CheckpointSaverHook):
#     def __init__(self, checkpoint_dir,
#             save_secs=None,
#             save_steps=None,
#             saver=None,
#             checkpoint_basename='model.ckpt',
#             scaffold=None,
#             listeners=None,
#             save_last_steps=None,
#         ):
# 
#         self.count = 0
#         super().__init__(checkpoint_dir, save_secs, save_steps, saver, 
#                          checkpoint_basename, scaffold, listeners)
#         
#     def before_run(self, run_context):
#         if self.count > 25:
#             self._timer._every_steps = 1
#         self.count += 1
#         return super().before_run(run_context)
#     
# estimator.fit(input_fn=train_input_fn, monitors=[CheckpointHook("/tmp/a", save_steps=10)])
# ```
# 
# **RunConfig**
# 
# - [Tensorflow Github: tf.estimator.RunConfig](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/estimator/run_config.py)
# 
# ```python
# # default configuration
# config = tf.estimator.RunConfig(
#     model_dir=None,
#     tf_random_seed=None,
#     save_summary_steps=100,
#     save_checkpoints_steps=100,
#     save_checkpoints_secs=600,
#     session_config=None,
#     keep_checkpoint_max=5,
#     keep_checkpoint_every_n_hours=10000,
#     log_step_count_steps=100,
# )
# 
# svm.create_linear_model(args.learning_rate, input_dim, config=config)
# svm.create_rffm_model(args.learning_rate, input_dim, args.dimension, args.stddev, config=config)
# ```
# 

# In[ ]:


def config(args):
    return tf.estimator.RunConfig(
        model_dir=args.model_dir,
        tf_random_seed=None,
        save_summary_steps=100,
        save_checkpoints_steps=None,
        save_checkpoints_secs=86400,
        session_config=None,
        keep_checkpoint_max=args.max_checkpoint,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=100,
    )


# In[ ]:


def main(args):
    
    # Loading Dataset and Preprocessing Data
    X_train, Y_train, X_test, Y_test, train_sample, valid_sample = svm.load(args.window_size)

    # Build Input Fn
    train_input_fn = svm.np_input_fn(X_train, 
                                 Y_train, 
                                 samples=train_sample, 
                                 shuffle=True, 
                                 window_size=args.window_size,
                                 batch=args.batch,
                                 epoch=args.epoch)

    # Training Model
    input_dim = args.window_size * args.window_size * 6
    if args.model == "linear":
        estimator = svm.create_linear_model(args.learning_rate, input_dim, config=config(args))
    if args.model == "rffm":
        estimator = svm.create_rffm_model(args.learning_rate, input_dim, args.dimension, args.stddev, config=config(args))
        
    start = time.time()
    if args.train:
        estimator.fit(input_fn=train_input_fn) # Train.
    train_sec = time.time() - start
    print('Training Elapsed time: {} seconds'.format(train_sec))
    
    # Evaluating Training Data
    if not args.evaluate:
        return
    
    start = time.time()
    train_metrics = svm.evaluate_model(estimator, X_train, Y_train, batch=65536, samples=train_sample, window_size=args.window_size)
    valid_metrics = svm.evaluate_model(estimator, X_train, Y_train, batch=65536, samples=valid_sample, window_size=args.window_size)
    testing_metrics = svm.evaluate_model(estimator, X_test, Y_test, batch=65536, window_size=args.window_size)
    eval_sec = time.time() - start
    print('Evaluate Elapsed time: {} seconds'.format(eval_sec))
    
    print(train_metrics)
    print(valid_metrics)
    print(testing_metrics)
    
    global_step = estimator.get_variable_value("global_step")
    result = "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %s %f %f %f\n" % (
          train_metrics["tp"], train_metrics["fp"], train_metrics["fn"], train_metrics["tn"],
          valid_metrics["tp"], valid_metrics["fp"], valid_metrics["fn"], valid_metrics["tn"],
          testing_metrics["tp"], testing_metrics["fp"], testing_metrics["fn"], testing_metrics["tn"], 
          global_step, args.epoch, args.batch, args.window_size, args.learning_rate, args.dimension, args.stddev,
          args.model, train_sec, eval_sec, train_sec + eval_sec)
    print(result)

    f = open(args.output,"a+")
    f.write(result)
    f.close()


# In[ ]:


if __name__ == "__main__":
    class Args:
        # mode
        train = True
        evaluate = True
        
        # cache
        model = "linear" # or "rffm"
        model_dir = None
        max_checkpoint = 10
        
        # train
        batch = 2048
        epoch = 2
        window_size = 23

        # Optimizer
        learning_rate = 0.001

        # RFFM
        dimension = 31740
        stddev = 1.0

        # Output
        output = "a.out"

    main(Args())


# In[ ]:




