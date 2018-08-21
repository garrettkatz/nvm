import numpy as np
from learning_rules import *
from preprocessing import preprocess

def unique(x):
    return list(set(x))

def assemble(nvmnet, programs, verbose=False, orthogonal=False, other_tokens=[]):

    registers = nvmnet.devices.keys()

    ### Preprocess program strings
    lines, labels, tokens = preprocess(programs, registers)
    all_labels = set()
    for name in labels: all_labels |= set(labels[name].keys())
    all_tokens = list(tokens.union(other_tokens).union(all_labels))

    ### Encode instruction pointers and labels
    ip_tokens = []
    name_offsets = {}
    for name in lines:
        name_offsets[name] = len(ip_tokens)
        index_width = str(int(np.ceil(np.log10(len(lines[name])))))
        ip_tokens.extend(
            [name] + [("%s.%0"+index_width+"d") % (name,l) for l in range(len(lines[name]))])

    ip_patterns = nvmnet.layers["ip"].encode_tokens(tokens=ip_tokens, orthogonal=orthogonal)
    for name in lines:
        for label, line_index in labels[name].items():
            # index is + 1 for leading program name pointer, but - 1 for ip -> opx step
            nvmnet.layers["ip"].coder.encode(label,
                ip_patterns[:,[name_offsets[name]+line_index]])

    ### Track total number of errors
    weights, biases = {}, {}
    diff_count = 0

    ### Sequence ip
    if verbose: print("Sequencing ip -> ip")
    weights[("ip","ip")], biases[("ip","ip")], dc = flash_mem(
        np.zeros((ip_patterns.shape[0], ip_patterns.shape[0])),
        np.zeros((ip_patterns.shape[0], 1)),
        ip_patterns[:,:-1], ip_patterns[:,1:],
        nvmnet.layers["ip"].activator,
        nvmnet.layers["ip"].activator,
        nvmnet.learning_rules[("ip","ip")],
        verbose=verbose)
    diff_count += dc
    
    ### Link instructions to ip
    for i,x in enumerate("c12"):
        if verbose: print("Linking ip -> op"+x)
        weights[("op"+x,"ip")] = np.zeros((nvmnet.layers["op"+x].size, nvmnet.layers["ip"].size))
        biases[("op"+x,"ip")] = np.zeros((nvmnet.layers["op"+x].size, 1))
        for name in programs:
            encodings = nvmnet.layers["op"+x].encode_tokens(
                [line[i] for line in lines[name]])
            ip_patterns = nvmnet.layers["ip"].encode_tokens(
                ip_tokens[name_offsets[name]:name_offsets[name]+len(lines[name])])
            weights[("op"+x,"ip")], biases[("op"+x,"ip")], dc = flash_mem(
                weights[("op"+x,"ip")], biases[("op"+x,"ip")],
                ip_patterns, encodings,
                nvmnet.layers["ip"].activator,
                nvmnet.layers["op"+x].activator,
                nvmnet.learning_rules[("op"+x,"ip")],
                verbose=verbose)
            diff_count += dc

    ### Link tokens across pathways
    pathways = [(r1, r2) for r1 in registers for r2 in registers] # for movd
    pathways += [(r, "op2") for r in registers] # for movv
    pathways += [("ci", r) for r in registers] # for cmpd
    pathways += [("ci", "op2")] # for cmpv
    pathways += [("ip", r) for r in registers] # jmpd, subd
    pathways += [("ip", "op1")] # for jmpv, subv

    for pathway in pathways:
    
        # Set up training data
        to_name, from_name = pathway
        if verbose: print("Linking %s -> %s"%(from_name, to_name))
        to_layer = nvmnet.layers[to_name]
        from_layer = nvmnet.layers[from_name]
        common_tokens = list(
            set(to_layer.all_tokens()) & set(from_layer.all_tokens()))
        X = from_layer.encode_tokens(common_tokens)
        Y = to_layer.encode_tokens(common_tokens)
        
        # Learn associations
        weights[pathway], biases[pathway], dc = flash_mem(
            np.zeros((to_layer.size, from_layer.size)),
            np.zeros((to_layer.size, 1)),
            X, Y, from_layer.activator, to_layer.activator,
            nvmnet.learning_rules[pathway],
            verbose=verbose)
        diff_count += dc
        
    return weights, biases, diff_count
    
if __name__ == '__main__':

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

    layer_size = 64
    pad = 0.95
    devices = {}

    from nvm_net import NVMNet
    # activator, learning_rule = logistic_activator, logistic_hebbian
    activator, learning_rule = tanh_activator, tanh_hebbian
    
    nvmnet = NVMNet(layer_size, pad, activator, learning_rule, devices)

    program = """
    
start:  nop
        set r1 true
        mov r2 r1
        jmp cond start
        end
    """

    program_name = "test"    
    weights, biases = assemble(nvmnet, program, program_name, verbose=True)

    ip = nvmnet.layers["ip"]
    f = ip.activator.f
    jmp = False
    end = False
    v = nvmnet.layers["ip"].coder.encode(program_name)
    for t in range(20):
        line = ""
        for x in "c12":
            opx = nvmnet.layers["op"+x]
            o = opx.activator.f(weights[("op"+x,"ip")].dot(v) + biases[("op"+x,"ip")])
            line += " " + opx.coder.decode(o)
            if x == 'c' and opx.coder.decode(o) == "jmp": jmp = True
            if x == 'c' and opx.coder.decode(o) == "end": end = True
        v = f(weights[("ip","ip")].dot(v) + biases[("ip","ip")])
        line = ip.coder.decode(v) + " " + line
        print(line)
        if end: break
        if jmp:
            v = nvmnet.layers["op2"].coder.encode("start")    
            v = f(weights[("ip","op2")].dot(v) + biases[("ip","op2")])
            jmp = False

    print(ip.coder.decode(v))
