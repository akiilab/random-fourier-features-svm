import os
os.environ['ROOT'] = os.path.dirname(os.path.realpath(__file__))

train_data_dict = {
        'fcsv_phs': [
            './cases/data/3x5.csv',
            './cases/data/5x10.csv',
            './cases/data/100x200.csv',
            ],
        'tcsv_phs': [
            './cases/target/3x5.csv',
            './cases/target/5x10.csv',
            './cases/target/100x200.csv',
            ],

        }

eval_data_dict = {
        'fcsv_phs': [
            './cases/data/3x4.csv',
            './cases/data/10x20.csv',
            ],
        'tcsv_phs': [
            './cases/target/3x4.csv',
            './cases/target/10x20.csv',
            ],
        }

train_dataset_dict = {
        'Short-TrainSet-UdrSamp-3_3_1p0_1p0_0p1': [
            './cases/sample/3x5.pkl',
            './cases/sample/5x10.pkl',
            './cases/sample/100x200.pkl',
            ],
        }

valid_dataset_dict = {
        'Short-ValidSet-NoUdrSamp': [
            './cases/sample/3x4.pkl',
            './cases/sample/10x20.pkl',
            ],
        }
