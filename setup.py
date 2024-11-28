import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

linetrace = os.environ.get("LINETRACE", False) == "True"

ext_modules = [
    Extension(
        "pyempad_calibrate.utils",
        ["src/pyempad_calibrate/utils.pyx"],
    )
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={"language_level": "3", "linetrace": linetrace},
    ),
    include_dirs=[np.get_include()],
)
