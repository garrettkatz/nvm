# nvm
`nvm` implements a Neural Virtual Machine (NVM).  This is a neural network that emulates a symbolic machine using distributed representation and local learning.  Human-authored assembly programs for the symbolic machine can be represented and executed by the neural network.

## Requirements

`nvm` has been tested using the following environment, but it may work with other operating systems and versions.
* [Fedora](https://getfedora.org/) 29
* [Python](https://www.python.org/) 2.7.15
* [numpy](http://www.numpy.org/) 1.15.4
* [scipy](http://www.scipy.org/scipylib/index.html) 1.0.0
* [matplotlib](http://matplotlib.org/) 2.2.3

## Installation

1. [Clone or download](https://help.github.com/articles/cloning-a-repository/) this repository into a directory of your choice.
2. Add the `src` sub-directory to your [PYTHONPATH](https://docs.python.org/2/using/cmdline.html#envvar-PYTHONPATH).

## Basic Usage

First, decide on the registers you want for your NVM instance.  For example:

```
>>> register_names = ["r0", "r1"]
```

Begin by writing programs in NVM assembly.  Each program should be given a name and stored in a dictionary with its name as the key.  For example, the following program implements and invokes a sub-routine for logical and, using 

