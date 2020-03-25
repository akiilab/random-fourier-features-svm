#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import fio
import svm
import numpy as np
import itertools


# In[ ]:


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test, train_sample, valid_sample, testing_sample = svm.load(23)


# In[ ]:


def count(Y, sample):
    if isinstance(Y, list):
        arr = np.array([], dtype=int)
        for y, s in zip(Y, sample):
            arr = np.append(arr, count(y, s))
        return arr.reshape((-1,3))

    if sample is None:
        return np.array([0,0,0])

    Y = Y[sample[:,0], sample[:,1]]
    total = Y.shape[0]
    p = np.count_nonzero(Y)
    n = total - p

    return np.array([total, p, n])


# In[ ]:


if __name__ == "__main__":
    train = count(Y_train, train_sample)
    vaild = count(Y_train, valid_sample)
    test  = count(Y_test, testing_sample)

    print("idx %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s" % ("train", "p", "n", "valid", "p", "n", "test", "p", "n"))
    for i, (a, b, c) in enumerate(itertools.zip_longest(list(train), list(vaild), list(test), fillvalue=np.array([0,0,0]))):
        print("%2s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s" % (i, a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]))
    print("-- ----------- ----------- ----------- ----------- ----------- ----------- ----------- ----------- -----------")

    a, b, c = np.sum(train, axis=0), np.sum(vaild, axis=0), np.sum(test, axis=0)
    print("sum %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s" % (a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]))


# In[ ]:


def sum_stat(Y, sample):
    return np.sum(count(Y, sample), axis=0)


# In[ ]:


if __name__ == "__main__":
    train = sum_stat(Y_train, train_sample)
    print(train)


# In[ ]:




