#!/usr/bin/env python
import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

exts = [
    Extension(
        name="matcha.utils.monotonic_align.core",
        sources=["matcha/utils/monotonic_align/core.pyx"],
    )
]

setup(
    ext_modules=cythonize(exts, language_level=3),
    include_dirs=[numpy.get_include()],
)
