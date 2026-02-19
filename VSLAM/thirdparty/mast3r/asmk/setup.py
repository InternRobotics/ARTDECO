import sys
from setuptools import setup, Extension
from setuptools.command.install import install

setup(
    ext_modules=[Extension("asmk.hamming", ["cython/hamming.c"])],
)
