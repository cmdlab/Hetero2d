# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='hetero2d', 
    version='1.0.1',
    platform='linux',
    description='Automation workflow to simulate the synthesis of two dimensional materials',  
    long_description=long_description,  
    long_description_content_type='text/x-rst',  
    url='https://github.com/cmdlab/Hetero2d', 
    author='Tara M. Boland. Lead Developer', 
    author_email='tara.boland01@gmail.com',  

    classifiers=[  
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Academic Research',
        'Topic :: Computational :: DFT :: Automation',
        'License :: GNU GPL',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='synthesis density-functional-theory two-dimensional 2D automation pymatgen',  

    packages=find_packages(),
    python_requires='>=3.6, <4',
    install_requires=[],  
    project_urls={'Bug Reports': 'https://github.com/cmdlab/Hetero2d/issues',
        'Source': 'https://github.com/cmdlab/Hetero2d/',
    },
)
