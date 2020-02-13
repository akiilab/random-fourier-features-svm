import argparse
import fio
import time
import preprocessing as pc
import main as m
from config import *

def arg_parse():
    parser = argparse.ArgumentParser(description='Process some integers.')

    # cache
    parser.add_argument('-m', '--model-dir', default=None, type=str, help='cache model dir')

    # train
    parser.add_argument('-b', '--batch', default=2048, type=int, help='batch size')
    parser.add_argument('-e', '--epoch', default=2, type=int, help='epoch')
    parser.add_argument('-w', '--window-size', default=23, type=int, help='window size')

    # Optimizer
    parser.add_argument('-l', '--learning-rate', default=0.001, type=float, help='Adam Optimizer learning rate')

    # RFFM
    parser.add_argument('-d', '--dimension', default=31740, type=int, help='Ramdom Fouier Features Mapper output dimension')
    parser.add_argument('-s', '--stddev', default=1.0, type=int, help='Ramdom Fouier Features Mapper standard deviation distribution')

    # Output
    parser.add_argument('-o', '--output', default="a.out", type=str, help='Output file')

    args = parser.parse_args()
    return args


def train(X, Y, stat, args, train_sample=[]):
    epoch = args.epoch
    batch = args.batch
    w = args.window_size

    # Cache
    model_dir = args.model_dir

    # Adam Optimizor
    l = args.learning_rate

    # RFFM
    input_dim = w * w * 6
    output_dim = args.dimension
    stddev = args.stddev

    print(input_dim)

    estimator = m.create_model(l, input_dim, output_dim, stddev, model_dir)

    X, Y = m.preprocessing(X, Y, stat, window_size=w, samples=train_sample)
    m.train_model(estimator, X, Y, batch, epoch)

    return estimator

def main():
    args = arg_parse()

    X_train_orig = fio.load_file(X_train_dataset)
    Y_train_orig = fio.load_file(Y_train_dataset)
    X_test_orig = fio.load_file(X_test_dataset)
    Y_test_orig = fio.load_file(Y_test_dataset)
    train_sample = fio.load_sample_file(train_sample_dataset)
    valid_sample = fio.load_sample_file(valid_sample_dataset)
    stat = pc.get_feat_stat(X_train_orig)

    start = time.time()

    estimator = train(X_train_orig, Y_train_orig, stat, args, train_sample)
    global_step = estimator.get_variable_value("global_step")
    print("global step:", global_step)

    train_stat   = m.evaluate_models(estimator, X_train_orig, Y_train_orig, stat, window_size=args.window_size, batch=args.batch, num_thread=1, samples=train_sample)
    print(train_stat)
    valid_stat   = m.evaluate_models(estimator, X_train_orig, Y_train_orig, stat, window_size=args.window_size, batch=args.batch, num_thread=1, samples=valid_sample)
    print(valid_stat)
    testing_stat = m.evaluate_models(estimator, X_test_orig,  Y_test_orig,  stat, window_size=args.window_size, batch=args.batch, num_thread=1)
    print(testing_stat)

    end = time.time()
    print('Elapsed time: {} seconds'.format(end - start))

    result = "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %f\n" % (
          train_stat["tp"], train_stat["fp"], train_stat["fn"], train_stat["tn"],
          valid_stat["tp"], valid_stat["fp"], valid_stat["fn"], valid_stat["tn"],
          testing_stat["tp"], testing_stat["fp"], testing_stat["fn"], testing_stat["tn"],
          global_step, args.epoch, args.batch, args.window_size, args.learning_rate, args.dimension, args.stddev, end-start)
    print(result)

    f = open(args.output,"w+")
    f.write(result)
    f.close()

if __name__ == "__main__":
    main()

