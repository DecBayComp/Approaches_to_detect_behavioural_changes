'''setup.py is needed to install in editable mode'''

from setuptools import setup
from setuptools.extension import Extension
import Cython.Build
import os

if __name__ == '__main__':
    extensions = [Extension("maggotuba.mmd.cmmd", [os.path.join("src", "maggotuba", "mmd", "cmmd.pyx")])]
    setup(ext_modules = Cython.Build.cythonize(extensions, annotate=False))