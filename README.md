# FUNFOLDING [![Build Status](https://travis-ci.com/tudo-astroparticlephysics/funfolding.svg?branch=master)](https://travis-ci.com/tudo-astroparticlephysics/funfolding) [![PyPI version](https://badge.fury.io/py/funfolding.svg)](https://badge.fury.io/py/funfolding)

```
         _____
     _-~~     ~~-_//
   /~             ~\
  |              _  |_
 |         _--~~~ )~~ )___
\|        /   ___   _-~   ~-_
\          _-~   ~-_         \
|         /         \         |
|        |           |     (O  |
 |      |             |        |
 |      |   O)        |       |
 /|      |           |       /
 / \ _--_ \         /-_   _-~)
   /~    \ ~-_   _-~   ~~~__/
  |   |\  ~-_ ~~~ _-~~---~  \
  |   | |    ~--~~  / \      ~-_
   |   \ |                      ~-_
    \   ~-|                        ~~--__ _-~~-,
     ~-_   |                             /     |
        ~~--|                                 /
          |  |                               /
          |   |              _            _-~
          |  /~~--_   __---~~          _-~
          |  \                   __--~~
          |  |~~--__     ___---~~
          |  |      ~~~~~
          |  |

```

Python library to perform spectral unfoldings.

## Getting Started



### Prerequisites

The prerequesites are handled by the `setup.py`, if you are using conda, you might
want to install the available packages via `conda` first.
```
matplotlib
numpy
pymc3
scikit-learn>=0.18.1
scipy
six
```

### Installing

Install via pip:

```
pip install funfolding
```

## Running the tests

The package has some basic unit tests. You can test them by

```
python setup.py test

```

Another way to test the installation is to run the examples from the examples folder.
