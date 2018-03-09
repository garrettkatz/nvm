import numpy as np
from learning_rules import *

def link(nvmnet, verbose=False):
    """Link token encodings between layer pairs"""

    weights, biases = {}, {}

    # link op2 layer with device layers for set instruction
    op2 = nvmnet.layers["op2"]
    tokens = op2.coder.list_tokens()
    X = np.concatenate(map(op2.coder.encode, tokens), axis=1)
    for name, layer in nvmnet.devices.items():
        Y = np.concatenate(map(layer.coder.encode, tokens), axis=1)
        weights[(name, "op2")], biases[(name, "op2")] = flash_mem(
            X, Y, layer.activator, nvmnet.learning_rule, verbose=verbose)
    
    return weights, biases

if __name__ == '__main__':

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

    from nvm_net import make_nvmnet
    nvmnet = make_nvmnet()

    weights, biases = link(nvmnet, verbose=True)

    op2 = nvmnet.layers["op2"]
    d1 = nvmnet.layers["d1"]
    w, b = weights["d1","op2"], biases["d1","op2"]

    print(d1.coder.decode(d1.activator.f(w.dot(op2.coder.encode("true")) + b)))
    print(d1.coder.decode(d1.activator.f(w.dot(op2.coder.encode("null")) + b)))
    print(d1.coder.encode("null").T)
    print(op2.coder.encode("null").T)
