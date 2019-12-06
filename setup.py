#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
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

from setuptools import setup

import os.path

# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = ["numpy", "scipy", "attrs", "matplotlib"]

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md")) as fp:
    LONG_DESCRIPTION = fp.read()

DESCRIPTION = "Grid Search in Python"


# =============================================================================
# FUNCTIONS
# =============================================================================

def do_setup():
    setup(
        name="grispy",
        version="0.0.1",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',

        author=[
            "Martin Chalela",
            "Emanuel Sillero",
            "Luis Pereyra",
            "Alejandro Garcia"],
        author_email="tinchochalela@gmail.com",
        url="https://github.com/mchalela/GriSPy",
        license="MIT",

        keywords=["grispy", "nearest", "neighbors", "search", "grid"],

        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering"],

        packages=["grispy"],
        py_modules=["ez_setup"],

        install_requires=REQUIREMENTS,

    )


if __name__ == "__main__":
    do_setup()
