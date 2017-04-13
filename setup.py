"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(

    name='bmicp',
    version='1.2.0',
    description='biomodule - iterative calculation of stomatal pores ',
    long_description=long_description,
    url='https://github.com/totti0223/',
    author='Yosuke Toda',
    author_email='tyosuke@aquaseerser.com',
    license='MIT',

    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        #'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
    keywords='automatic quantification of stomatal pores',
    packages=['bmicp'],
    install_requires=open('requirements.txt').read().splitlines(),
)
