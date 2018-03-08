import numpy as np
from learning_rules import *

def assemble(nvmnet, program, name, learning_rule, verbose=False):

    ### Preprocess program string

    labels = dict() # map label to line number
    # split up lines and remove blanks
    lines = [line.strip() for line in program.splitlines() if len(line.strip()) > 0]
    for l in range(len(lines)):
        # remove comments
        comment = lines[l].find("#")
        if comment > -1: lines[l] = lines[l][:comment]
        # split out tokens
        lines[l] = lines[l].split()
        # check for label
        if lines[l][0][-1] == ":":
            # remove and save label
            labels[lines[l][0][:-1]] = l
            lines[l] = lines[l][1:]
        # pad with nulls
        while len(lines[l]) < 4:
            lines[l].append("null")

    ### Encode instruction pointers and labels
    ips = [nvmnet.layers["ip"].coder.encode(name)] # pointer to program
    for l in range(len(lines)):
        ips.append(nvmnet.layers["ip"].coder.encode("ip %2d"%l))
    for label, line_index in labels.items():
        ip_index = line_index + 1 # + 1 for leading program pointer
        ip_index -= 1 # but - 1 for ip -> opx step
        nvmnet.layers["ip"].coder.encode(label, ips[ip_index])
    ips = np.concatenate(ips, axis=1)

    ### Encode tokens and replace labels
    encodings = {"op"+x:list() for x in "c123"}
    for l in range(len(lines)):
        for o,x in enumerate("c123"):
            pattern = nvmnet.layers["op"+x].coder.encode(lines[l][o])
            encodings["op"+x].append(pattern)
    encodings = {k: np.concatenate(v,axis=1) for k,v in encodings.items()}
    
    ### Store tokens
    weights, biases = {}, {}
    for x in "c123":
        weights[("op"+x,"ip")], biases[("op"+x,"ip")] = flash_mem(
            ips[:,:-1], encodings["op"+x],
            nvmnet.layers["op"+x].activator,
            learning_rule, verbose=verbose)

    ### Store instruction pointer sequence
    weights[("ip","ip")], biases[("ip","ip")] = flash_mem(
        ips[:,:-1], ips[:,1:],
        nvmnet.layers["ip"].activator,
        learning_rule, verbose=verbose)

    ### Store label links
    if len(labels) > 0:
        label_tokens = labels.keys()
        X_label = np.concatenate([
            nvmnet.layers["op2"].coder.encode(tok)
            for tok in label_tokens], axis=1)
        Y_label = np.concatenate([
            nvmnet.layers["ip"].coder.encode(tok)
            for tok in label_tokens], axis=1)
        weights[("ip","op2")], biases[("ip","op2")] = flash_mem(
            X_label, Y_label,
            nvmnet.layers["ip"].activator,
            learning_rule, verbose=verbose)
    
    return weights, biases

def flash_mem(X, Y, activator, learning_rule, verbose=False):
    
    w, b = learning_rule(X, Y, activator)

    if verbose:
        _Y = activator.f(w.dot(X) + b)
        print("Flash ram residual max: %f"%np.fabs(Y - _Y).max())
        print("Flash ram residual mad: %f"%np.fabs(Y - _Y).mean())
        print("Flash ram sign diffs: %d"%((np.ones(Y.shape) - activator.e(Y, _Y)).sum()))

    return w, b

if __name__ == '__main__':

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

    layer_size = 64
    pad = 0.95
    devices = {}

    from nvm_net import NVMNet
    nvmnet = NVMNet(layer_size, pad, devices)

    program = """
    
start:  nop
        set r1 true
        mov r2 r1
        jmp cond start
        end
    """

    program_name = "test"    
    weights, biases = assemble(nvmnet, program, program_name, logistic_hebbian, verbose=True)


    ip = nvmnet.layers["ip"]
    f = ip.activator.f
    jmp = False
    end = False
    v = nvmnet.layers["ip"].coder.encode(program_name)
    for t in range(20):
        line = ""
        for x in "c123":
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
