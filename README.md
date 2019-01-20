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

Next, write some NVM assembly programs for the instance.  Each program should be given a name and stored in a dictionary with its name as the key.  For example, the following program implements and invokes a sub-routine that computes logical and of the two register contents, using the primitive comparison instruction:

```
>>> programs = {
"myfirstprogram":"""

### computes logical-and of r0 and r1, overwriting r0 with result

        nop           # do nothing
        sub and       # call logical-and sub-routine
        exit          # halt execution

and:    cmp r0 false  # compare first conjunct to false
        jie and.f     # jump, if equal to false, to and.f label
        cmp r1 false  # compare second conjunct to false
        jie and.f     # jump, if equal false, to and.f label
        mov r0 true   # both conjuncts true, set r0 to true
        ret           # return from sub-routine
and.f:  mov r0 false  # a conjunct was false, set r0 to false
        ret           # return from sub-routine

"""}
```

Now we can construct an NVM instance to run this program.  We will use a convenience method that automatically scales the network size to accommodate the program.

```
>>> from nvm.nvm import make_scaled_nvm
>>> my_nvm = make_scaled_nvm(
...     register_names = register_names,
...     programs = programs,
...     orthogonal=True)

```

The `orthogonal=True` flag uses orthogonal activity patterns to represent symbols, which greatly reduces the size requirements.  Now we can assemble the program into the instance, and load it so that it is ready to be executed.  When loading we can also specify initial activity for each register when execution begins.

```
>>> my_nvm.assemble(programs)
>>> my_nvm.load("myfirstprogram",
...     initial_state = {"r0":"true","r1":"false"})
```

The underlying representations for `true` and `false` are distributed activity patterns that we can access from the `net` field of the `nvm` instance.  Each pattern is a `numpy` array.

```
>>> my_nvm.net.layers["r0"].shape
(8, 1)
>>> my_nvm.net.activity["r0"].T
[[ 0.9999  0.9999  0.9999 -0.9999  0.9999  0.9999 -0.9999  0.9999]]
```

Each layer has a `coder` we can use to look-up the human-readable symbol represented by a pattern:

```
>>> v = my_nvm.net.activity["r0"].T
>>> my_nvm.net.layers["r0"].coder.decode(v)
true
```

The `nvm` instance also has convenience wrappers for this:

```
>>> my_nvm.decode_state(layer_names=register_names)
{'r0': 'true', 'r1': 'false'}
```

The program is already loaded; all we need to do to emulate it is run the network dynamics until an `exit` opcode is reached:

```
>>> import itertools
>>> for t in itertools.count():
...     my_nvm.net.tick()
...     if my_nvm.at_exit(): break

```

The invocation `my_nvm.at_exit()` is a convenience wrapper for:
```
>>> (my_nvm.net.layers["opc"].coder.decode(my_nvm.net.activity["opc"]) == "exit")
True
```

We can see how many time-steps of network dynamics were used:
```
>>> t
106
```

We can check the final register states when the program is finished:
```
>>> my_nvm.decode_state(layer_names=register_names)
{'r0': 'false', 'r1': 'false'}
```

`r0` was correctly overwritten with the logical-and of the initial register values, `true` and `false`.

