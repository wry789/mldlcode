# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
setup(ext_modules = cythonize(Extension(
    'glove_build_cooccurance',  # 生成的动态链接库的名字
    sources=['BuildCooccuranceMatrix.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))


import numpy as np
import math
def buildWeightMatrix_apply(element,xmax):
    return math.pow(element/xmax,0.75) if element >=xmax else 1
xmax=100.0
b=np.array([[100,101,22],[999,222,22]])
buildWeightMatrix_apply_np = np.vectorize(buildWeightMatrix_apply)
buildWeightMatrix_apply_np(b,xmax)