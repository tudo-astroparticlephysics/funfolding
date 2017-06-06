from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='funfolding',
    version='0.0.1',

    description='Having fun with unfolding.',
    long_description=long_description,

    url='https://github.com/mbrner/funfolding',

    author='Mathis Boerner',
    author_email='mathis.boerner@tu-dortmund.de',

    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.5',
    ],
    keywords='unfolding',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'numpy',
        'scikit-learn>=0.18.1',
        'emcee',
        'scipy',
        'matplotlib',
        'corner'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
