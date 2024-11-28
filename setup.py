import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "pyempad_calibrate.utils",
        ["src/pyempad_calibrate/utils.pyx"],
    )
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "language_level": "3",
            "linetrace": os.environ.get("LINETRACE", False) == "True",
        },
    ),
    include_dirs=[np.get_include()],
)
