"""Crosswinds: A 2D + 1D Cross-correlation code for Astronomy

See:
https://github.com/mubdi/crosswinds-python
"""

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "crosswinds/crosswinds",
        ["crosswinds/crosswinds.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()],
        depends=['cross_corr_2d_1d.c']
    )
]

setup(
    name='crosswinds',
    version='0.0.1.dev1',
    description='A cross-correlation code for position-position-velocity cubes',
    url='https://github.com/mubdi/crosswinds-python',
    author='Mubdi Rahman',
    author_email='mubdi@jhu.edu',
    license='Apache',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Programming Languages :: Python :: 2',
        'Programming Languages :: Python :: 3',
        'Programming Language :: C',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering :: Astronomy'
    ],
    packages=find_packages(),
    keywords='astronomy cross-correlation',
    install_requires=['numpy', 'cython'],
    ext_modules=cythonize(ext_modules)
)
