# setup.py
import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("utils.node", ["utils/node.pyx"])
]

setup(
    name="cython_node",
    ext_modules=cythonize(extensions),
)