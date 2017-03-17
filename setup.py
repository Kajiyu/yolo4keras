# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name = "yolo4keras",
    version = "0.1.0",
    url = "https://github.com/Kajiyu/yolo4keras",
    description = "Yolo implementation for Keras",
    author = "Kaji",
    license = 'MIT',
    classifiers = [
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
    ],
    install_requires = [
            'numpy',
            'tensorflow',
            'keras',
            ],
    keywords = 'object-detecting',
    packages = find_packages(),
    packages = ['yolo4keras'],
)
