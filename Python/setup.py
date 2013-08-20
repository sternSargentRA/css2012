import os
import numpy as np

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

incdirs = [os.path.join(os.getcwd(), 'SimpleRNG'),
           os.path.join(os.getcwd(), 'src'),
           np.get_include()]

css2012 = Extension("css2012CythonFuncs", ["css2012CythonFuncs.pyx"],
                    include_dirs=incdirs, language="c++")

# Define extensions
xdress_extras = Extension("rng.xdress_extra_types",
                          ["rng/xdress_extra_types.pyx"],
                          include_dirs=incdirs, language="c++")


stl_cont = Extension("rng.stlcontainers", ["rng/stlcontainers.pyx"],
                     include_dirs=incdirs, language="c++")


SimpleRNG = Extension("rng.SimpleRNG",
                      ["rng/SimpleRNG.pyx", "src/SimpleRNG.cpp"],
                      include_dirs=incdirs, language="c++")


ext_modules = [stl_cont, xdress_extras, css2012, SimpleRNG]

setup(cmdclass={'build_ext': build_ext},
      name="cssCy",
      version='1.0',
      ext_modules=ext_modules,
      )
