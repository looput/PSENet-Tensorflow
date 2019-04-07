# "-std=gnu++11"
# NOTE: https://stackoverflow.com/questions/45653307/cython-undefined-symbol-with-c-wrapper/45655654#45655654
# https://www.zhihu.com/question/23003213/answer/56121859
# https://stackoverflow.com/questions/29168575/wrap-c-class-with-cython-getting-the-basic-example-to-work

import cython

import numpy as np
cimport numpy as np

cdef extern from "my_lib.h" namespace "MY_LIB":
    cdef cppclass Expand:
        Expand() except +
        # int* CC, Si
        void expansion(int*,int*,int,int)
    
    cdef cppclass Region:
        Region() except +
        void region_grow(double* ,int*,int* ,int* , double ,int*,double*, int &)

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

# Test passed
cdef class PyRegion:
    cdef Region *thisptr
    def __cinit__(self):
        self.thisptr = new Region()
    def __dealloc__(self):
        del self.thisptr
    def region_grow(self,np.ndarray[double, ndim=3,mode="c"] embed_vc not None,\
        np.ndarray[int, ndim=2,mode="c"] mask not None,\
        np.ndarray[int, ndim=1,mode="c"] seed not None, delta,\
        np.ndarray[int, ndim=2,mode="c"] flags not None,
        np.ndarray[double,ndim=1,mode='c'] ins_vec not None):

        cdef double delta_c=delta
        cdef int num
        cdef np.ndarray[int, ndim=1] shape = np.array([embed_vc.shape[0],embed_vc.shape[1],embed_vc.shape[2]],dtype=np.int32)
        # cdef np.ndarray[double,ndim=1] ins_vec=np.array([embed_vc.shape[2]],dtype=np.float64)

        self.thisptr.region_grow(<double *> np.PyArray_DATA(embed_vc),\
                <int*> np.PyArray_DATA(mask),\
                <int*> np.PyArray_DATA(shape),\
                <int*> np.PyArray_DATA(seed),\
                delta_c,\
                <int*> np.PyArray_DATA(flags),\
                <double*> np.PyArray_DATA(ins_vec),\
                num)
        return num