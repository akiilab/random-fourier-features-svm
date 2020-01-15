import numpy as np

# under sampling
def undersample(arr, idx):
    # filt = np.load('./cases/sample/3x5.pkl', allow_pickle=True)
    # filt = np.array(filt)
    # filt[:,[0, 1]] = filt[:,[1, 0]]  # reverse
    # X_train[0]['data'][filt[:,1],filt[:,0]]
    if isinstance(arr, list):
        return [undersample(f, i) for f, i in zip(arr, idx)]
    return arr[idx[:,1],idx[:,0]]

# def get_feat_stat(feat_map):
#     feat_stats = {}
#     feat_sum_list = []
#     feat_squ_sum_list = []
#     max_val_list = []
#     obs_cnt_list = []
# 
#     for dsgn in feat_map:
#         # fm = feat_map[dsgn]
#         fm = dsgn
# 
#         feat_sum_list.append(np.sum(fm, axis=(0,1), dtype=np.float128))
#         feat_squ_sum_list.append(np.sum(np.square(fm), axis=(0,1), dtype=np.float128))
#         max_val_list.append(np.max(fm, axis=(0,1)))
#         obs_cnt_list.append(fm.shape[0] * fm.shape[1])
# 
#     total_obs_cnt = np.sum(obs_cnt_list)
#     mean = np.divide(np.sum(feat_sum_list, axis=0), total_obs_cnt)
#     squ_mean = np.divide(np.sum(feat_squ_sum_list, axis=0), total_obs_cnt)
#     var = squ_mean - np.square(mean)
#     stdev = np.sqrt(var)
#     max_val = np.max(max_val_list, axis=0)
# 
#     feat_stats = {
#             'MaxVal': max_val,
#             'ExpVal': mean,
#             'StdDev': stdev,
#             'Var': var
#             }
# 
#     return feat_stats

def get_feat_stat(arr):
    arr = [ f.reshape((-1, f.shape[-1])) for f in arr ]
    arr = np.concatenate(arr)

    return {
            'max': np.max(arr, axis=0),
            'mean': np.mean(arr, axis=0, dtype=np.float128),
            'stdev': np.nanstd(arr, axis=0, dtype=np.float128),
            'var': np.nanvar(arr, axis=0, dtype=np.float128),
            }

# https://stackoverflow.com/a/4544459
def standardize(A, stat):
    # A = (A - mean(A, axis=0)) / std(A, axis=0)
    if isinstance(A, list):
        return [ standardize(a, stat) for a in A ]

    A = np.subtract(A, stat['mean'])
    A = np.divide(A, stat['stdev'])
    A = A.astype(np.float32)

    return A
    # A = np.where(np.isfinite(std_out), std_out, np.zeros_like(std_out))

    # for fm in A:
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         std_out = np.divide(np.subtract(fm, exp_val), std_dev)
    #         ppc_feat_map = np.where(np.isfinite(std_out), std_out, np.zeros_like(std_out))

def expand(A, window_size):
    if not window_size & 0x1:
        raise Exception('need odd value on padding')

    if isinstance(A, list):
        return [ expand(a, window_size) for a in A ]

    # For example
    # A is a (5,3,6) matrix
    # window_size is 5
    # 
    # A = np.arange(0,5*3*6,1).reshape((5,3,6)).astype(np.float128)
    # A = np.pad(A, ((2,2), (2,2), (0,0)), mode='constant')
    # A = strided(A, shape=(5,3,5,5,6), strides=(672,96,672,96,16))
    # A = A.reshape((5,3,150))
    #
    # For more information:
    # https://zhuanlan.zhihu.com/p/64933417

    n = window_size # the height and width of the window
    p = window_size >> 1 # the padding size

    d0, d1, d2 = A.shape # dimansion 0, 1, 2
    s0, s1, s2 = A.strides # stride 0, 1, 2

    A = np.pad(A, pad_width=((p,p),(p,p),(0,0)), mode='constant')
    A = np.lib.stride_tricks.as_strided(A, shape=(d0,d1,n,n,d2), strides=(s0,s1,s0,s1,s2))
    A = A.reshape((d0,d1,d2*n*n))

    return A

def classify(A):
    if isinstance(A, list):
        return [ classify(a) for a in A ]

    A = A != 0
    A = A.astype(int)

    return A
