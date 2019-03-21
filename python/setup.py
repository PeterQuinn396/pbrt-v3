from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(name = "TestClass",
      ext_modules=cythonize("TestClass.pyx"), include_path = [numpy.get_include()],
      language = 'c++'
      )