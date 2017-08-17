from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os

ext_module = Extension(
    name="calc_score_naive_bayes",
    sources=[
        os.path.join(
            "calc_score_naive_bayes.pyx"
        )
    ],
    extra_compile_args=['-O3', '-ffast-math', '-march=native'],
)

setup(
    name='calc_core_naive_bayes app',
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_module],
    include_dirs=[np.get_include()]
)
