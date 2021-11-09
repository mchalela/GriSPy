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

"""This file manages the distribution and installation of GriSPy."""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib

from setuptools import setup

# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = ["numpy", "scipy", "attrs"]

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

with open(PATH / "README.md") as fp:
    LONG_DESCRIPTION = fp.read()

with open(PATH / "grispy" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break


DESCRIPTION = "Grid Search in Python"


# =============================================================================
# FUNCTIONS
# =============================================================================


def do_setup():
    setup(
        name="grispy",
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=[
            "Martin Chalela",
            "Emanuel Sillero",
            "Luis Pereyra",
            "Alejandro Garcia",
            "Juan B. Cabral",
        ],
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
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering",
        ],
        packages=["grispy"],
        install_requires=REQUIREMENTS,
    )


if __name__ == "__main__":
    do_setup()
