# "-std=gnu++11"
# NOTE: https://stackoverflow.com/questions/45653307/cython-undefined-symbol-with-c-wrapper/45655654#45655654
# https://www.zhihu.com/question/23003213/answer/56121859
# https://stackoverflow.com/questions/29168575/wrap-c-class-with-cython-getting-the-basic-example-to-work

import cython

import numpy as np
cimport numpy as np

cdef extern from "Expand.h" namespace "PSE":
    cdef cppclass Expand:
        Expand() except +
        # int* CC, Si
        void expansion(int*,int*,int,int)

cdef class PyExpand:
    cdef Expand *thisptr
    def __cinit__(self):
        self.thisptr = new Expand()
    def __dealloc__(self):
        del self.thisptr
    def expansion(self,np.ndarray[int, ndim=2,mode="c"] CC not None,np.ndarray[int, ndim=2,mode="c"] Si not None):
        cdef int m, n
        h, w = CC.shape[0], CC.shape[1]
        self.thisptr.expansion(<int *> np.PyArray_DATA(CC),\
                <int*> np.PyArray_DATA(Si),h,w)