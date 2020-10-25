#!python
#cython: language_level=3

import numpy as np
import sys
import time
cimport numpy as np
cimport cython

cdef extern from "Python.h":
    const char* PyUnicode_AsUTF8(object unicode)

from libc.stdlib cimport malloc, free
from libc.string cimport strcmp

cdef char ** to_cstring_array(list_str):
    cdef char **ret = <char **>malloc(len(list_str) * sizeof(char *))
    for i in xrange(len(list_str)):
        ret[i] = PyUnicode_AsUTF8(list_str[i])
    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef emission_matrix(obs_seq,sta_seq):
    """
    计算发射矩阵
    """
    cdef char **c_arr1 = to_cstring_array(obs_seq)
    cdef char **c_arr2 = to_cstring_array(sta_seq)
    cdef Py_ssize_t i,j
    cdef int k,n
    cdef np.ndarray os,ss,obs_space,states_space,o
    cdef int[:, :] ef

    os = np.array(list(obs_seq)) # 字序列
    ss = np.array(list(sta_seq)) # 状态序列
    obs_space = np.unique(os) # 字集合
    states_space = np.unique(ss) # 状态集合
    k = states_space.size 
    n = obs_space.size
    ef = np.zeros((k, n),dtype=np.intc) # 发射矩阵
    
    # 此处可用cython加速
    for i in range(k): # ['B', 'E', 'M', 'S']
        print(f"now state is {i}")
        for j in range(n):# ['举', '乃', '久', '么', '义'...]
            # states_space:状态集合 ['B', 'E', 'M', 'S'] ,ss:状态序列 ，os:字序列
            # 状态i(B)在状态序列中的位置索引，从观测序列中取出同位置的字
            # i状态对应的所有字
            o = os[ss == states_space[i]] 
            ef[i, j] = np.count_nonzero(o == obs_space[j]) # obs_space：字集合，状态i到j字的个数
            
    ep = ef / ef.sum(axis=1)[:, None]
    return ep