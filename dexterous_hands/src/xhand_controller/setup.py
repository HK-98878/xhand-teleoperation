#!/usr/bin/env python

from distutils.core import setup
from os.path import abspath, basename, dirname

from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=[basename(dirname(abspath(__file__)))],
    package_dir={"": "src"},
)

setup(**setup_args)
