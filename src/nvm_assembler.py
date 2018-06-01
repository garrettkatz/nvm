import numpy as np
from learning_rules import *
from preprocessing import preprocess
from nvm_encoder import encode_tokens

def assemble(nvmnet, program, name, verbose=False, orthogonal=False):

    ### Preprocess program string
    lines, labels = preprocess(program, nvmnet.devices.keys())

    ### Encode tokens
    tokens = list(set(tok for line in lines for tok in line))
    encode_tokens(nvmnet, tokens, verbose, orthogonal)

    ### Encode instruction pointers and labels
    index_width = str(int(np.ceil(np.log10(len(lines)))))
    ips = [nvmnet.layers["ip"].coder.encode(name)] # pointer to program
    for l in range(len(lines)):
        ips.append(nvmnet.layers["ip"].coder.encode(
            ("%s.%0"+index_width+"d") % (name,l)))
    for label, line_index in labels.items():
        ip_index = line_index + 1 # + 1 for leading program pointer
        ip_index -= 1 # but - 1 for ip -> opx step
        nvmnet.layers["ip"].coder.encode(label, ips[ip_index])
    ips = np.concatenate(ips, axis=1)

    ### Encode tokens in op layers
    encodings = {"op"+x:list() for x in "c12"}
    for l in range(len(lines)):
        # encode ops
        for o,x in enumerate("c12"):
            pattern = nvmnet.layers["op"+x].coder.encode(lines[l][o])
            encodings["op"+x].append(pattern)
    encodings = {k: np.concatenate(v,axis=1) for k,v in encodings.items()}

    ### Track total number of errors
    diff_count = 0
    
    ### Bind op tokens to instruction pointers
    weights, biases = {}, {}
    for x in "c12":
        if verbose: print("Binding ip -> op"+x)
        weights[("op"+x,"ip")], biases[("op"+x,"ip")], dc = flash_mem(
            np.zeros((encodings["op"+x].shape[0], ips.shape[0])),
            np.zeros((encodings["op"+x].shape[0], 1)),
            ips[:,:-1], encodings["op"+x],
            nvmnet.layers["ip"].activator,
            nvmnet.layers["op"+x].activator,
            nvmnet.learning_rules[("op"+x,"ip")],
            verbose=verbose)
        diff_count += dc

    ### Store instruction pointer sequence
    if verbose: print("Binding ip -> ip"+x)
    weights[("ip","ip")], biases[("ip","ip")], dc = flash_mem(
        np.zeros((ips.shape[0], ips.shape[0])),
        np.zeros((ips.shape[0],1)),
        ips[:,:-1], ips[:,1:],
        nvmnet.layers["ip"].activator,
        nvmnet.layers["ip"].activator,
        nvmnet.learning_rules[("ip","ip")],
        verbose=verbose)
    diff_count += dc

    ### Bind labels to instruction pointers
    if len(labels) > 0:
        label_tokens = labels.keys()
        X_label = np.concatenate([
            nvmnet.layers["op2"].coder.encode(tok)
            for tok in label_tokens], axis=1)
        Y_label = np.concatenate([
            nvmnet.layers["ip"].coder.encode(tok)
            for tok in label_tokens], axis=1)
        if verbose: print("Binding op2 -> ip")
        weights[("ip","op2")], biases[("ip","op2")], dc = flash_mem(
            np.zeros((Y_label.shape[0], X_label.shape[0])),
            np.zeros((Y_label.shape[0], 1)),
            X_label, Y_label,
            nvmnet.layers["op2"].activator,
            nvmnet.layers["ip"].activator,
            nvmnet.learning_rules[("ip","op2")],
            verbose=verbose)
        diff_count += dc
    
    return weights, biases, dc

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
