from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import multiprocessing
import os
import sys


package_name = "cryptanalysislib"
packages = find_packages()

def readfile(filename):
    with open(filename,  encoding='utf-8') as f:
        return f.read()

setup(
    name=package_name,
    version="1.0.0",
    author="PingFloyd",
    description="break stuff",
    packages=packages,
    package_dir={package_name: package_name},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
