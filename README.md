# FUNFOLDING [![Build Status](https://travis-ci.com/tudo-astroparticlephysics/funfolding.svg?branch=master)](https://travis-ci.com/tudo-astroparticlephysics/funfolding)

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

You will need the development version of emcee, install using

```
pip install https://github.com/dfm/emcee/archive/master.tar.gz
```


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
pip install funfolding
```

## Running the tests

The package has some basic unit tests. You can test them by

```
python setup.py test

```

Another way to test the installation is to run the examples from the examples folder.
