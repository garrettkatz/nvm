import numpy as np

def assemble(nvmnet, program, name, do_global=False, verbose=False):

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
    ips = []
    for l in range(len(lines)):
        ips.append(nvmnet.layers["ip"].coder.encode("ip %2d"%l))
    for label, line_index in labels.items():
        nvmnet.layers["ip"].coder.encode(label, ips[line_index])
    ips.insert(0, nvmnet.layers["ip"].coder.encode(name)) # pointer to program
    ips = np.concatenate(ips, axis=1)

    ### Encode tokens and replace labels
    encodings = {"op"+x:list() for x in "c123"}
    for l in range(len(lines)):
        for o,x in enumerate("c123"):
            pattern = nvmnet.layers["op"+x].coder.encode(lines[l][o])
            encodings["op"+x].append(pattern)
    encodings = {k: np.concatenate(v,axis=1) for k,v in encodings.items()}
    
    ### Store tokens
    weights, bias = {}, {}
    for x in "c123":
        weights[("op"+x,"ip")], bias["op"+x] = flash_mem(
            ips[:,:-1], encodings["op"+x], nvmnet.layers["op"+x].activator,
            do_global=do_global, verbose=verbose)

    ### Set up unbiased instruction sequence
    X_ip = np.concatenate((
        ips[:,:-1], # pointers
        np.zeros((nvmnet.layers["op2"].size, ips.shape[1]-1)), # no label bias
        ), axis=0)
    Y_ip = ips[:,1:] # pointers at next step

    ### Set up label links
    if len(labels) > 0:
        label_tokens = labels.keys()
        label_op_patterns = np.concatenate([
            nvmnet.layers["op2"].coder.encode(tok)
            for tok in label_tokens])
        label_ip_patterns = np.concatenate([
            nvmnet.layers["ip"].coder.encode(tok)
            for tok in label_tokens])
        X_label = np.concatenate((
            np.zeros((nvmnet.layers["ip"].size, len(labels))), # no ip bias
            label_op_patterns, # op bias
            ), axis=0)
        Y_label = label_ip_patterns
    else:
        X_label = np.zeros((X_ip.shape[0],0))
        Y_label = np.zeros((Y_ip.shape[0],0))
    
    # Store ip associations
    W, b = flash_mem(
        np.concatenate((X_ip, X_label), axis=1),
        np.concatenate((Y_ip, Y_label), axis=1),
        nvmnet.layers["ip"].activator,
        do_global=do_global, verbose=verbose)
    weights[("ip","ip")] = W[:,:nvmnet.layers["ip"].size]
    weights[("ip","op2")] = W[:,nvmnet.layers["ip"].size:]
    bias["ip"] = b
    
    return weights, bias

def flash_mem(X, Y, activator, do_global=False, verbose=False):
    
    # y = f(Wx)
    # W = g(y)/x
    # W ~ g(y)x.T

    if do_global:
        W = np.linalg.lstsq(
            np.concatenate((X.T, np.ones((X.shape[1],1))), axis=1), # ones for bias
            activator.g(Y).T, rcond=None)[0].T
    else:
        W = activator.g(Y).dot(
            np.concatenate((X.T, np.ones((X.shape[1],1))),axis=1) # ones for bias
            ) / X.shape[0]

    weights, bias = W[:,:-1], W[:,[-1]]

    if verbose:
        _Y = activator.f(weights.dot(X) + bias)
        print("Flash ram residual max: %f"%np.fabs(Y - _Y).max())
        print("Flash ram residual mad: %f"%np.fabs(Y - _Y).mean())
        print("Flash ram sign diffs: %d"%((np.ones(Y.shape) - activator.e(Y, _Y)).sum()))

    return weights, bias

if __name__ == '__main__':

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

    layer_size = 8
    pad = 0.9
    devices = {}

    from nvm_net import NVMNet
    nvmnet = NVMNet(layer_size, pad, devices)

    program = """
    
start:  nop
        set r1 true
        mov r2, r1

    """
    name = "test"
    
    weights, bias = assemble(nvmnet, program, name, do_global=True, verbose=True)

    v = nvmnet.layers["ip"].coder.encode(name)

    ip = nvmnet.layers["ip"]
    f = ip.activator.f
    b = bias["ip"]
    for t in range(5):
        line = ""
        for x in "c123":
            opx = nvmnet.layers["op"+x]
            o = opx.activator.f(weights[("op"+x,"ip")].dot(v) + bias["op"+x])
            line += " " + opx.coder.decode(o)
        v = f(weights[("ip","ip")].dot(v) + bias["ip"])
        line = ip.coder.decode(v) + " " + line
        print(line)
