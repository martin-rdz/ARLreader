#!/usr/bin/python3

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("ARLreader/fast_funcs.pyx"),
    include_dirs=[numpy.get_include()]
)
