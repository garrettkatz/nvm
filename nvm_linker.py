import numpy as np
from learning_rules import *

def link(nvmnet, tokens=[], verbose=False):
    """Link token encodings between layer pairs"""

    weights, biases, diff_count = {}, {}, 0

    # include any constants and value/label tokens from op2
    op2 = nvmnet.layers["op2"]
    all_tokens = tokens + op2.coder.list_tokens() + nvmnet.constants

    # link op2 layer with device layers for set instruction
    X = np.concatenate(map(op2.coder.encode, all_tokens), axis=1)
    for name, layer in nvmnet.devices.items():
        Y = np.concatenate(map(layer.coder.encode, all_tokens), axis=1)
        if verbose: print("Linking op2 -> %s"%(name))
        weights[(name, "op2")], biases[(name, "op2")], dc = flash_mem(
            X, Y, layer.activator, nvmnet.learning_rule, verbose=verbose)
        diff_count += dc

    # link device layers with each other for mov instruction
    for from_name, from_layer in nvmnet.devices.items():
        X = np.concatenate(map(from_layer.coder.encode, all_tokens), axis=1)
        for to_name, to_layer in nvmnet.devices.items():
            Y = np.concatenate(map(to_layer.coder.encode, all_tokens), axis=1)
            if verbose: print("Linking %s -> %s"%(from_name, to_name))
            weights[(to_name, from_name)], biases[(to_name, from_name)], dc = flash_mem(
                X, Y, to_layer.activator, nvmnet.learning_rule, verbose=verbose)
            diff_count += dc    

    return weights, biases, diff_count

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
