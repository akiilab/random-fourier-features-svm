#!/bin/bash

# usage: run.py [-h] [--train] [--no-train] [--evaluate] [--no-evaluate]
#               [-m MODEL] [-p MODEL_DIR] [-b BATCH] [-e EPOCH] [-w WINDOW_SIZE]
#               [-l LEARNING_RATE] [-d DIMENSION] [-s STDDEV] [-o OUTPUT]
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
#                         path for caching model (default: None)
#   -b BATCH, --batch BATCH
#                         batch size (default: 2048)
#   -e EPOCH, --epoch EPOCH
#                         epoch (default: 2)
#   -w WINDOW_SIZE, --window-size WINDOW_SIZE
#                         window size (default: 23)
#   -l LEARNING_RATE, --learning-rate LEARNING_RATE
#                         Adam Optimizer learning rate (default: 0.001)
#   -d DIMENSION, --dimension DIMENSION
#                         Ramdom Fouier Features Mapper output dimension
#                         (default: 31740)
#   -s STDDEV, --stddev STDDEV
#                         Ramdom Fouier Features Mapper standard deviation
#                         distribution (default: 1.0)
#   -o OUTPUT, --output OUTPUT
#                         Output file

python3 run.py -e 20 -o kernel.out -m "./cache/model/kernel" # 20

