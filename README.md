# SVM Kernel Approximation

- Python 3.5
- TensorFlow 1.4.1

## Install

[Tensorflow r1.4 Doc](https://github.com/tensorflow/docs/blob/r1.4/site/en/install/install_mac.md#installing-with-anaconda)

```shell
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.4.1-py3-none-any.whl
pip install --ignore-installed --upgrade $TF_BINARY_URL
```

## Generate test cases

There are some test cases created by running the **./generate.sh** command. These data in test cases are randomly generated by **generate.py** file in order to fit the dataset on compute node. The default path of output file is under *./cache/case* directory.

```shell
./generate.sh
```

## Configurate the data set

Update the path of dataset in **config.py** file generated from above command.

```python
# config.py
train_data_dict = {
        'fcsv_phs': [ './cases/data/3x5.csv', './cases/data/5x10.csv' ],
        'tcsv_phs': [ './cases/target/3x5.csv', './cases/target/5x10.csv' ],
        }
eval_data_dict = {
        'fcsv_phs': [ './cases/data/3x4.csv' ],
        'tcsv_phs': [ './cases/target/3x4.csv' ],
        }

train_dataset_dict = {
        'Short-TrainSet-UdrSamp-3_3_1p0_1p0_0p1': [
            './cases/train_sample/3x5.pkl',
            './cases/train_sample/5x10.pkl',
          ],
        }
valid_dataset_dict = {
        'Short-ValidSet-NoUdrSamp': [
            './cases/valid_sample/3x5.pkl',
            './cases/valid_sample/5x10.pkl',
          ],
        }
```

## Run

```shell
$ python3 run.py -h
usage: run.py [-h] [--train] [--no-train] [--evaluate] [--no-evaluate]
              [-m MODEL] [-p MODEL_DIR] [-c MAX_CHECKPOINT] [-b BATCH]
              [-e EPOCH] [-w WINDOW_SIZE] [-l LEARNING_RATE] [-d DIMENSION]
              [-s STDDEV] [-o OUTPUT]

Process some integers.

optional arguments:
  -h, --help            show this help message and exit
  --train               enable training mode (default)
  --no-train            disable training mode
  --evaluate            enable evaluating mode (default)
  --no-evaluate         disable evluating mode
  -m MODEL, --model MODEL
                        using which model: "rffm", "linear" (default: "rffm")
  -p MODEL_DIR, --model-dir MODEL_DIR
                        directory where model parameters, graph, etc are
                        saved. If `None`, will use a default value set by the
                        Estimator. (default: None)
  -c MAX_CHECKPOINT, --max-checkpoint MAX_CHECKPOINT
                        The maximum number of recent checkpoint files to keep.
                        As new files are created, older files are deleted. If
                        0, all checkpoint files are kept. Defaults to 10 (that
                        is, the 10 most recent checkpoint files are kept.)
  -b BATCH, --batch BATCH
                        The number of consecutive elements of this dataset to
                        combine in a single batch. (default: 2048)
  -e EPOCH, --epoch EPOCH
                        the number of times the elements of this dataset
                        should be repeated (default: 2)
  -w WINDOW_SIZE, --window-size WINDOW_SIZE
                        window size (default: 23)
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        Adam Optimizer learning rate (default: 0.001)
  -d DIMENSION, --dimension DIMENSION
                        Ramdom Fouier Features Mapper. the output dimension of
                        the mapping. (default: 31740)
  -s STDDEV, --stddev STDDEV
                        Ramdom Fouier Features Mapper. The standard deviation
                        of the Gaussian kernel to be approximated. The error
                        of the classifier trained using this approximation is
                        very sensitive to this parameter.standard deviation
                        distribution (default: 1.0)
  -o OUTPUT, --output OUTPUT
                        statistic file
```

