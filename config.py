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
            './cases/train_sample/3x5.pkl',
            './cases/train_sample/5x10.pkl',
            './cases/train_sample/100x200.pkl',
            ],
        }

valid_dataset_dict = {
        'Short-ValidSet-NoUdrSamp': [
            './cases/valid_sample/3x5.pkl',
            './cases/valid_sample/5x10.pkl',
            './cases/valid_sample/100x200.pkl',
            ],
        }

X_train_dataset = train_data_dict['fcsv_phs']
Y_train_dataset = train_data_dict['tcsv_phs']
X_test_dataset  = eval_data_dict['fcsv_phs']
Y_test_dataset  = eval_data_dict['tcsv_phs']

train_sample_dataset = train_dataset_dict['Short-TrainSet-UdrSamp-3_3_1p0_1p0_0p1']
valid_sample_dataset = valid_dataset_dict['Short-ValidSet-NoUdrSamp']

