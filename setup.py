from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "pyempad_calibrate.utils",
        ["src/pyempad_calibrate/utils.pyx"],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()],
)
