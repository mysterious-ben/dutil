import pathlib
import os
from setuptools import setup, find_packages

PATH = pathlib.Path(__file__).parent
VERSION = os.getenv('VERSION', 'dev')
LONG_DESCRIPTION = (PATH/"README.md").read_text()

setup(
    name='dutil',
    version=VERSION,
    description='A few useful tools for data wrangling',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author='Yaroslav Kopotilov',
    license='Apache License, Version 2.0',
    author_email='UNKNOWN',
    url='https://github.com/mysterious-ben/dutil',
    install_requires=[
        'numpy',
        'pandas',
        'pyarrow',
        'dill',
        'loguru',
    ],
    entry_points={'console_scripts': []},
    packages=find_packages(),
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.7',
    test_suite='tests',
    setup_requires=[
        'pytest-runner',
        'setuptools',
    ],
    tests_require=[
        'pytest',
    ]
)


# --- pip upload ---
# python setup.py sdist bdist_wheel
# twine check dist/*
# twine upload dist/*
