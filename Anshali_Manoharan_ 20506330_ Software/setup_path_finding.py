# setup.py
import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("modules.path_finding_cy_dynamic_training", ["modules/path_finding_cy_dynamic_training.pyx"],
              include_dirs=[np.get_include()])
]

setup(
    name="pathfindingdynamictraining",
    ext_modules=cythonize(extensions),
)