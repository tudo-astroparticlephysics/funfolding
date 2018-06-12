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
corner
emcee
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
pip install https://github.com/tudo-astroparticlephysics/funfolding/archive/master.tar.gz
```

## Running the tests

The package has some basic unit tests. You can test them by

```
python setup.py test

```

Another way to test the installation is to run the examples from the examples folder.
