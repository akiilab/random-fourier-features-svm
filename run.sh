#!/bin/bash

# usage: run.py [-h] [-b BATCH] [-e EPOCH] [-w WINDOW_SIZE] [-l LEARNING_RATE]
#               [-d DIMENSION] [-s STDDEV] [-o OUTPUT]
# 
# Process some integers.
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   -m MODEL_DIR, --model-dir MODEL_DIR
#                         cache model dir
#   -b BATCH, --batch BATCH
#                         batch size
#   -e EPOCH, --epoch EPOCH
#                         epoch
#   -w WINDOW_SIZE, --window-size WINDOW_SIZE
#                         window size
#   -l LEARNING_RATE, --learning-rate LEARNING_RATE
#                         Adam Optimizer learning rate
#   -d DIMENSION, --dimension DIMENSION
#                         Ramdom Fouier Features Mapper output dimension
#   -s STDDEV, --stddev STDDEV
#                         Ramdom Fouier Features Mapper standard deviation
#                         distribution
#   -o OUTPUT, --output OUTPUT
#                         Output file

python3 run.py -e 1 -o kernel.out -m "./cache/model/kernel" # 1
python3 run.py -e 1 -o kernel.out -m "./cache/model/kernel" # 2
python3 run.py -e 1 -o kernel.out -m "./cache/model/kernel" # 3
python3 run.py -e 1 -o kernel.out -m "./cache/model/kernel" # 4
python3 run.py -e 1 -o kernel.out -m "./cache/model/kernel" # 5
python3 run.py -e 5 -o kernel.out -m "./cache/model/kernel" # 10
python3 run.py -e 5 -o kernel.out -m "./cache/model/kernel" # 15
python3 run.py -e 5 -o kernel.out -m "./cache/model/kernel" # 20
