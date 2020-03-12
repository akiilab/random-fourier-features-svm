#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import fio
import svm
import numpy as np
import itertools


# In[ ]:


def load():
    X_train, Y_train, X_test, Y_test, train_sample, valid_sample = svm.load(1)
    testing_sample = [np.indices((x.shape[0], x.shape[1])).reshape((2,-1)).T for x in X_test]
    return X_train, Y_train, X_test, Y_test, train_sample, valid_sample, testing_sample


# In[ ]:


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test, train_sample, valid_sample, testing_sample = load()


# In[ ]:


def count(Y, sample):
    if sample is None:
        return np.array([0,0,0]), np.array([str(0), str(0), str(0)])
    Y = Y[sample[:,0], sample[:,1]]
    total = Y.shape[0]
    p = np.count_nonzero(Y)
    n = total - p
    return np.array([total, p, n]), np.array([str(total), str(p), str(n)])

train = np.array([], dtype=int)
vaild = np.array([], dtype=int)
test  = np.array([], dtype=int)
for i, x in enumerate(itertools.zip_longest(Y_train, Y_test, train_sample, valid_sample, testing_sample)):
    a, b, c = count(x[0], x[2]), count(x[0], x[3]), count(x[1], x[4])
    
    train = np.append(train, a[0])
    vaild = np.append(vaild, b[0])
    test  = np.append(test,  c[0])

train = train.reshape((-1,3))
vaild = vaild.reshape((-1,3))
test  = test.reshape((-1,3))

values = np.concatenate([train, vaild, test], axis=1)

print("idx %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s" % ("train", "p", "n", "valid", "p", "n", "test", "p", "n"))
i = 0
for a in values:
    print("%2s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s" % (i, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]))
    i += 1
print("-- ----------- ----------- ----------- ----------- ----------- ----------- ----------- ----------- -----------")
a = np.sum(values, axis=0)
print("sum %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s" % (a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]))


# In[ ]:




