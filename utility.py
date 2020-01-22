#!/usr/bin/env python
# coding: utf-8

# # Utility
# 
# This file is used to write some functions that standard library didn't have.

# ## Human readable file size
# 
# a function that convert to human readable size from bytes size.
# 
# https://stackoverflow.com/a/1094933

# In[ ]:


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


# In[ ]:


# usage
if __name__ == "__main__":
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()), key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

