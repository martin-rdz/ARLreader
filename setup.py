#!/usr/bin/python3

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("ARLreader/fast_funcs.pyx")
    #ext_modules = cythonize("ARLreader/fast_funcs.pyx", compiler_directives={'linetrace': True, 'binding': True})
)