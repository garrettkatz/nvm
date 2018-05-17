import numpy as np
from learning_rules import *

def link(nvmnet, tokens=[], verbose=0):
    """Link token encodings between layer pairs"""

    weights, biases, diff_count = {}, {}, 0

    # list all tokens
    ip = nvmnet.layers["ip"]
    op1 = nvmnet.layers["op1"]
    op2 = nvmnet.layers["op2"]
    all_tokens = set()
    all_tokens.update(tokens)
    all_tokens.update(ip.coder.list_tokens())
    all_tokens.update(op1.coder.list_tokens())
    all_tokens.update(op2.coder.list_tokens())
    all_tokens.update(op2.coder.list_tokens())
    all_tokens.update(nvmnet.constants)

    # link op2 layer with device layers for movv instruction
    for name, layer in nvmnet.devices.items():
        X = np.concatenate(map(op2.coder.encode, all_tokens), axis=1)
        Y = np.concatenate(map(layer.coder.encode, all_tokens), axis=1)
        if verbose > 0: print("Linking op2 -> %s"%(name))
        weights[(name, "op2")], biases[(name, "op2")], dc = flash_mem(
            np.zeros((Y.shape[0],X.shape[0])),
            np.zeros((Y.shape[0],1)),
            X, Y, op2.activator, layer.activator,
            nvmnet.learning_rules[(name,"op2")],
            verbose=verbose)
        diff_count += dc

    # link device layers with each other for movd instruction
    for from_name, from_layer in nvmnet.devices.items():
        for to_name, to_layer in nvmnet.devices.items():
            if from_name == to_name: continue
            X = np.concatenate(map(from_layer.coder.encode, all_tokens), axis=1)
            Y = np.concatenate(map(to_layer.coder.encode, all_tokens), axis=1)
            if verbose > 0: print("Linking %s -> %s"%(from_name, to_name))
            pair_key = (to_name, from_name)
            weights[pair_key], biases[pair_key], dc = flash_mem(
                np.zeros((Y.shape[0],X.shape[0])),
                np.zeros((Y.shape[0],1)),
                X, Y, from_layer.activator, to_layer.activator,
                nvmnet.learning_rules[pair_key],
                verbose=verbose)
            diff_count += dc    

    # link device layers to ci for cmpd instruction
    ci = nvmnet.layers["ci"]
    for name, layer in nvmnet.devices.items():
        X = np.concatenate(map(layer.coder.encode, all_tokens), axis=1)
        Y = np.concatenate(map(ci.coder.encode, all_tokens), axis=1)
        if verbose > 0: print("Linking %s -> ci"%(name))
        weights[("ci", name)], biases[("ci", name)], dc = flash_mem(
            np.zeros((Y.shape[0],X.shape[0])),
            np.zeros((Y.shape[0],1)),
            X, Y, layer.activator, ci.activator,
            nvmnet.learning_rules[("ci", name)],
            verbose=verbose)
        diff_count += dc

    # link op2 to ci for cmpv instruction
    X = np.concatenate(map(op2.coder.encode, all_tokens), axis=1)
    Y = np.concatenate(map(ci.coder.encode, all_tokens), axis=1)
    if verbose > 0: print("Linking op2 -> ci")
    weights[("ci", "op2")], biases[("ci", "op2")], dc = flash_mem(
        np.zeros((Y.shape[0],X.shape[0])),
        np.zeros((Y.shape[0],1)),
        X, Y, op2.activator, ci.activator,
        nvmnet.learning_rules[("ci", "op2")],
        verbose=verbose)
    diff_count += dc

    # link device layers to ip for jmpd, subd instructions
    for name, layer in nvmnet.devices.items():
        X = np.concatenate(map(layer.coder.encode, all_tokens), axis=1)
        Y = np.concatenate(map(ip.coder.encode, all_tokens), axis=1)
        if verbose > 0: print("Linking %s -> ip"%(name))
        weights[("ip", name)], biases[("ip", name)], dc = flash_mem(
            np.zeros((Y.shape[0],X.shape[0])),
            np.zeros((Y.shape[0],1)),
            X, Y, layer.activator, ip.activator,
            nvmnet.learning_rules[("ip", name)],
            verbose=verbose)
        diff_count += dc

    # link op1 to ip for jmpv, subv instructions
    X = np.concatenate(map(op1.coder.encode, all_tokens), axis=1)
    Y = np.concatenate(map(ip.coder.encode, all_tokens), axis=1)
    if verbose > 0: print("Linking op1 -> ip")
    weights[("ip", "op1")], biases[("ip", "op1")], dc = flash_mem(
        np.zeros((Y.shape[0],X.shape[0])),
        np.zeros((Y.shape[0],1)),
        X, Y, op1.activator, ip.activator,
        nvmnet.learning_rules[("ip","op1")],
        verbose=verbose)
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
