FUNFOLDING [![Build Status](https://travis-ci.org/mbrner/funfolding.svg?branch=master)](https://travis-ci.org/mbrner/funfolding)
===========

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

```
numpy
scikit-learn>=0.18.1
emcee
scipy
matplotlib
corner
```

Optional:

```
pymc
```

### Installing

The library can be simple check out from this github repo and installed via pip

```
git clone git@github.com:mbrner/funfolding.git
cd funfolding && pip install .
```

## Running the tests

The package has some basic unit tests. You can test them by

```
python setup.py test

```

Another way to test the installation is to run the examples from the examples folder.
