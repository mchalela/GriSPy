#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute and install GriSPy
"""


# =============================================================================
# IMPORTS
# =============================================================================

from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages  # noqa


# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = ["numpy"]

with open("README.md") as fp:
    LONG_DESCRIPTION = fp.read()

DESCRIPTION = "Grid Search in Python"


# =============================================================================
# FUNCTIONS
# =============================================================================

def do_setup():
    setup(
        name='grispy',
        version='2019.9',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author='Martin Chalela',
        author_email='tinchochalela@gmail.com',
        url='https://github.com/mchalela/GriSPy',
        license='MIT',

        keywords=['grispy', 'project', 'neighbors', 'grid'],

        classifiers=(
            'Development Status :: 4 - Beta',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: Implementation :: CPython',
            'Topic :: Scientific/Engineering'),

        py_modules=['grispy', 'ez_setup'],

        install_requires=REQUIREMENTS,

        # extras_require={  # Optional
        #    'example': ['example'],
        #    'test': ['pytest','coverage','pytest-cov'],
        # }
        )



if __name__ == "__main__":
    do_setup()