from learning_rules import *
from orthogonal_patterns import random_hadamard

def encode_tokens(nvmnet, tokens, verbose=False, orthogonal=False):
    T = len(tokens)
    registers = nvmnet.devices.keys()
    for layer_name in ["op1","op2","ci"] + registers:
        layer = nvmnet.layers[layer_name]
        if orthogonal:
            patterns = random_hadamard(layer.size, T)
            for t in range(T):
                layer.coder.encode(tokens[t], patterns[:,[t]])
        else:
            for t in range(T):
                layer.coder.encode(tokens[t])
    
    pathways = [(r1, r2) for r1 in registers for r2 in registers] # for movd
    pathways += [(r, "op2") for r in registers] # for movv
    pathways += [("ci", r) for r in registers] # for cmpd
    pathways += [("ci", "op2")] # for cmpv
    pathways += [("ip", r) for r in registers] # jmpd, subd
    pathways += [("ip", "op1")] # for jmpv, subv
    
    for pathway in pathways:
    
        # Set up training data
        to_name, from_name = pathway
        to_layer = nvmnet.layers[to_name]
        from_layer = nvmnet.layers[from_name]
        X = np.empty((from_layer.size, T))
        Y = np.empty((to_layer.size, T))
        for t in range(T):
            X[:,[t]] = from_layer.coder.encode(tokens[t])
            Y[:,[t]] = to_layer.coder.encode(tokens[t])
        
        # Learn associations
        weights = nvmnet.weights.get(pathway,
            np.zeros((to_layer.size, from_layer.size)))
        biases = nvmnet.biases.get(pathway,
            np.zeros((to_layer.size, 1)))
        weights, biases, dc = flash_mem(
            weights, biases,
            X, Y, from_layer.activator, to_layer.activator,
            nvmnet.learning_rules[pathway],
            verbose=verbose)
        nvmnet.weights[pathway] = weights
        nvmnet.biases[pathway] = biases
