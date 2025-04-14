# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 16:28:42 2025

@author: jpmog
"""

from setuptools import setup, find_packages

setup(
    name="nn_visualizer",
    version="0.1",
    description="Feedforward Neural Network Visualizer for Keras models",
    author="",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "tensorflow"
    ],
)
