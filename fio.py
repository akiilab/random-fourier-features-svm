import os
import hashlib
import numpy as np

# s, rows, cols = '"1.23 2.34"', [], []
# return: 0.0
# rows = [1.23]
# cols = [2.34]
def collect_coordinate(s, rows, cols):
    r, c = np.fromstring(s.strip(b"\""), dtype=np.float32, sep=' ')

    if len(rows) == 0 or r > rows[-1]: rows.append(r)
    if len(cols) == 0 or c > cols[-1]: cols.append(c)

    return np.float32(0.0)

def parse_csv(data_file):
    rows, cols = [], []
    conv = lambda s: collect_coordinate(s, rows, cols)

    with open(data_file, 'r') as f:
        arr = np.genfromtxt(f, delimiter=',', skip_header=1, dtype=np.float32, converters={0: conv})
        arr = arr[:,1:]
        arr = arr.reshape((len(cols), len(rows), -1), order='F')

    return {
            'data': arr,
            'rows': np.array(rows, dtype=np.float32),
            'cols': np.array(cols, dtype=np.float32),
            }

# return the real path of file
def abspath(file_name):
    return os.path.abspath(os.path.join(os.environ['ROOT'], file_name))

# return the md5 hash of file
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_file(data_file):
    if isinstance(data_file, (list)):
        return [load_file(f) for f in data_file]

    data_file = abspath(data_file)
    cache_file = os.path.join(os.environ['ROOT'], "cache/dataset/%s.npz" % md5(data_file))

    if not os.path.exists(os.path.dirname(cache_file)):
        os.makedirs(os.path.dirname(cache_file))

    if not os.path.isfile(cache_file):
        print('cache not found. save %s to %s' % (data_file, cache_file))
        data = parse_csv(data_file)
        np.savez(cache_file, **data)

    return np.load(cache_file)['data']

def load_sample_file(data_file):
    if isinstance(data_file, (list)):
        return [load_sample_file(f) for f in data_file]

    arr = np.load(data_file, allow_pickle=True)
    arr = np.array(arr)
    arr[:,[0, 1]] = arr[:,[1, 0]]

    return arr

