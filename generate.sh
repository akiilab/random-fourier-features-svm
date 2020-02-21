#!/bin/bash

# usage: generate.py [-h] [-o OUTPUT] [-d DIR] rows cols
# 
# Process some integers.
# 
# positional arguments:
#   rows                  the number of row
#   cols                  the number of column
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   -o OUTPUT, --output OUTPUT
#                         (option) output file. default `a`
#   -d DIR, --dir DIR     (option) output file directory. default `case`

python generate.py -o 3x4 3 4
python generate.py -o 3x5 3 5
python generate.py -o 5x10 5 10
python generate.py -o 10x20 10 20
python generate.py -o 100x200 100 200

