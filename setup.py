#!/usr/bin/env python

from setuptools import find_packages, setup

# get version
with open("climfill/version.py") as f:
    line = f.readline().strip().replace(" ", "").replace('"', "")
    version = line.split("=")[1]
    __version__ = version

    setup(
        name="climfill",
        version=__version__,
        description="filling missing values in multivariate, gridded, spatiotemporal data",
        author="Verena Bessenbacher",
        author_email="verena.bessenbacher@env.ethz.com",
        packages=find_packages(),
        url="https://github.com/climachine/climfill",
        install_requires=open("requirements.txt").read().split(),
        long_description="See README",
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.6",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Atmospheric Science",
            "Topic :: Scientific/Engineering :: Environmental Science",
            "Topic :: Scientific/Engineering :: Hydrology",
        ],
        python_requires=">=3.6",  # >=??
    )
