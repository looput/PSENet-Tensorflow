from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy

#  python setup.py build_ext --inplace
setup(ext_modules = cythonize(Extension(
           "mylib",                                # the extesion name
           sources=["my_lib.pyx","region_grow.cpp", "Expand.cpp"], # the Cython source and
                                                  # additional C++ source files
           language="c++",                        # generate and compile C++ code
           extra_compile_args=["-std=c++11"],      # must use C++11 or #error This file requires compiler and library support for the ISO C++ 2011 standard. This support must be enabled with the -std=c++11 or -std=gnu++11 compiler options
           include_dirs=[numpy.get_include()],
      )))