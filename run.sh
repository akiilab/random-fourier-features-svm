#!/bin/bash

# usage: run.py [-h] [--train] [--no-train] [--evaluate] [--no-evaluate]
#               [-m MODEL] [-p MODEL_DIR] [-c MAX_CHECKPOINT] [-b BATCH]
#               [-e EPOCH] [-w WINDOW_SIZE] [-l LEARNING_RATE] [-d DIMENSION]
#               [-s STDDEV] [-o OUTPUT]
# 
# Process some integers.
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   --train               enable training mode (default)
#   --no-train            disable training mode
#   --evaluate            enable evaluating mode (default)
#   --no-evaluate         disable evluating mode
#   -m MODEL, --model MODEL
#                         using which model: "rffm", "linear" (default: "rffm")
#   -p MODEL_DIR, --model-dir MODEL_DIR
#                         directory where model parameters, graph, etc are
#                         saved. If `None`, will use a default value set by the
#                         Estimator. (default: None)
#   -c MAX_CHECKPOINT, --max-checkpoint MAX_CHECKPOINT
#                         The maximum number of recent checkpoint files to keep.
#                         As new files are created, older files are deleted. If
#                         0, all checkpoint files are kept. Defaults to 10 (that
#                         is, the 10 most recent checkpoint files are kept.)
#   -b BATCH, --batch BATCH
#                         The number of consecutive elements of this dataset to
#                         combine in a single batch. (default: 2048)
#   -e EPOCH, --epoch EPOCH
#                         the number of times the elements of this dataset
#                         should be repeated (default: 2)
#   -w WINDOW_SIZE, --window-size WINDOW_SIZE
#                         window size (default: 23)
#   -l LEARNING_RATE, --learning-rate LEARNING_RATE
#                         Adam Optimizer learning rate (default: 0.001)
#   -d DIMENSION, --dimension DIMENSION
#                         Ramdom Fouier Features Mapper. the output dimension of
#                         the mapping. (default: 31740)
#   -s STDDEV, --stddev STDDEV
#                         Ramdom Fouier Features Mapper. The standard deviation
#                         of the Gaussian kernel to be approximated. The error
#                         of the classifier trained using this approximation is
#                         very sensitive to this parameter.standard deviation
#                         distribution (default: 1.0)
#   -o OUTPUT, --output OUTPUT
#                         statistic file

python3 run.py -e 200 -c 200 -o kernel.out -m "linear" -p "./cache/model/linear"

