import numpy as np
from tokens import get_token, N_LAYER, LAYERS, DEVICES, TOKENS
from gates import get_gates, get_open_gates, get_gate_index, PAD, LAMBDA, N_GATES, N_HGATES, N_GH
from flash_rom import W_ROM

#### FINALIZE WEIGHTS ###

# dict of inter/intra layer weight matrices
def make_weights():
    weights = {}
    # relays
    for to_layer in LAYERS + DEVICES:
        for from_layer in LAYERS + DEVICES:
            weights[(to_layer,from_layer)] = "relay" #np.eye(N_LAYER,N_LAYER) * LAMBDA
    # RAM (will be learned)
    weights[("MEM","MEM")] = "none"
    weights[("MEM","MEMH")] = np.zeros((N_LAYER, N_LAYER))
    weights[("MEMH","MEM")] = "none"
    weights[("MEMH","MEMH")] = np.zeros((N_LAYER, N_LAYER))
    # ROM
    weights[("GATES","GATES")] = W_ROM[:,:N_GH]
    weights[("GATES","OPC")] = W_ROM[:,N_GH+0*N_LAYER:N_GH+1*N_LAYER]
    weights[("GATES","OP1")] = W_ROM[:,N_GH+1*N_LAYER:N_GH+2*N_LAYER]
    weights[("GATES","OP2")] = W_ROM[:,N_GH+2*N_LAYER:N_GH+3*N_LAYER]
    weights[("GATES","OP3")] = W_ROM[:,N_GH+3*N_LAYER:N_GH+4*N_LAYER]
    return weights

def store_program(weights, program, do_global=False):
    # encode program transits
    V_prog = PAD*np.sign(np.concatenate(tuple(TOKENS[t] for t in program), axis=1)) # program
    V_prog = np.concatenate((V_prog, PAD*np.sign(np.random.randn(*V_prog.shape))),axis=0) # add hidden
    
    # link labels
    labels = {program[p]:p for p in range(0,len(program),5) if program[p] != "NULL"}
    for p in range(3,len(program),5):
        if program[p] in labels:
            V_prog[:N_LAYER,p+1] = V_prog[N_LAYER:, labels[program[p]]]
    
    # flash ram with program memory
    X, Y = V_prog[N_LAYER:,:-1], V_prog[:,1:]
    
    if do_global:
        # global
        W_ram = np.linalg.lstsq(X.T, np.arctanh(Y).T, rcond=None)[0].T
    else:
        # local
        # W_ram = np.arctanh(Y).dot(X.T) / N_LAYER #/ (N_LAYER*PAD**2)
        W_ram = np.zeros((2*N_LAYER, N_LAYER))
        for p in range(V_prog.shape[1]-1):
            W_ram = weight_update(W_ram, V_prog[N_LAYER:,[p]], V_prog[:,[p+1]])
    
    print("Flash ram residual max: %f"%np.fabs(Y - np.tanh(W_ram.dot(X))).max())
    print("Flash ram residual mad: %f"%np.fabs(Y - np.tanh(W_ram.dot(X))).mean())
    print("Flash ram sign diffs: %d"%(np.sign(Y) != np.sign(np.tanh(W_ram.dot(X)))).sum())
    if (np.sign(Y) != np.sign(np.tanh(W_ram.dot(X)))).sum() > 0:
        sys.exit(0)
    
    # ram
    weights[("MEM","MEMH")] = W_ram[:N_LAYER,:]
    weights[("MEMH","MEMH")] = W_ram[N_LAYER:,:]
    return weights, V_prog[:,[0]]


CTS_NOISE = 0.000*(1.-PAD)
FLIP_NOISE = 0.000

def state_string(activity):
    hr = ["%s:%s"%(k,get_token(activity[k])) for k in [
        "MEM","OPC","OP1","OP2","OP3","REG1","REG2","REG3","FEF","TC"
    ]]
    s = " ".join(hr)
    open_gates = get_open_gates(activity["GATES"])
    s += "\nopen gates: "
    s += ", ".join(["%s-%s-%s"%og for og in open_gates])
    return s

def print_state(activity):
    print(state_string(activity))

def tick(activity, weights):

    # NVM tick
    current_gates = get_gates(activity["GATES"])
    activity_new = {k: np.zeros(v.shape) for (k,v) in activity.items()}
    for (to_layer, from_layer) in weights:
        u = current_gates[(to_layer, from_layer, "U")]
        u = float(u > 0)
        w = weights[(to_layer, from_layer)]
        if type(w) == str and w == "none":
            wv = 0
        elif type(w) == str and w == "relay":
            wv = u * LAMBDA * activity[from_layer]
        else:
            wv = u * w.dot(activity[from_layer])
        if to_layer == from_layer:
            c = current_gates[(to_layer, from_layer, "C")]
            c = float(c > 0)
            wv += (1-u)*(1-c)*LAMBDA * activity[from_layer]
        activity_new[to_layer] += wv

    # handle compare specially, never gated
    cmp_e = 1./(2.*N_LAYER)
    w_cmph = np.arctanh(1. - cmp_e) / (PAD / 2.)**2
    w_cmpo = 2. * np.arctanh(PAD) / (N_LAYER*(1-cmp_e) - (N_LAYER-1))
    b_cmpo = w_cmpo * (N_LAYER*(1 - cmp_e) + (N_LAYER-1)) / 2.
    activity_new["CMPH"] = w_cmph * activity["CMPA"] * activity["CMPB"]
    activity_new["CMPO"] = np.ones((N_LAYER,1)) * (w_cmpo * activity["CMPH"].sum() - b_cmpo)
    
    for layer in activity_new:
        activity_new[layer] = np.tanh(activity_new[layer])
        # inject noise
        flip = (np.random.rand(activity_new[layer].shape[0]) < FLIP_NOISE)
        activity_new[layer][flip,:] = -activity_new[layer][flip,:]
        activity_new[layer] += np.random.randn(*activity_new[layer].shape)*CTS_NOISE

    return activity_new

def weight_update(W, x, y):
    return W + np.arctanh(y) * x.T / N_LAYER #/ (N_LAYER*PAD**2)

def nvm_synapto(weights):
    layers = [{
        "name" : "bias",
        "neural model" : "relay",
        "rows" : 1,
        "columns" : 1}]
    for layer in LAYERS + DEVICES + ["GATES"]:
        dendrites = []
        if layer not in ["CMPH", "CMPO"]:
            dendrites.extend(
                {"name" : layer + "<" + from_layer}
                    for (to_layer,from_layer),w in weights.iteritems()
                        if to_layer == layer and (type(w) is not str or (type(w) is str and w != "none"))
            )
            dendrites.append(
                {"name" : "gain",
                 "children" : [
                     {
                         "name" : "gain-update",
                         "opcode" : "add"
                     },
                     {
                         "name" : "gain-decay",
                         "opcode" : "mult"
                     }
                 ]
                }
            )
        layers.append({
            "name" : layer,
            "neural model" : "nvm",
            "dendrites" : dendrites,
            "rows" : 1 if layer == "GATES" else 32,
            "columns" : N_GH if layer == "GATES" else 32})
    structures = [{"name" : "nvm", "type" : "parallel", "layers": layers}]
    
    connections = []
    for (to_layer, from_layer),w in weights.iteritems():
        if to_layer in ["CMPH", "CMPO"] or from_layer in ["CMPH", "CMPA", "CMPB"]: continue
        if type(w) == str and w == "none": continue
        elif type(w) == str and w == "relay":
            # wv = u * LAMBDA * activity[from_layer]
            connections.append({
                "name": to_layer + "<"+  from_layer + "-input",
                "dendrite": to_layer + "<"+  from_layer,
                "from layer": from_layer,
                "to layer": to_layer,
                "type": "one to one",
                "opcode": "add",
                "plastic" : "false",
                "weight config" : {
                    "type" : "flat",
                    "weight" : LAMBDA
                },
            })
        else:
            # wv = u * w.dot(activity[from_layer])
            connections.append({
                "name": to_layer + "<"+  from_layer + "-input",
                "dendrite": to_layer + "<"+  from_layer,
                "from layer": from_layer,
                "to layer": to_layer,
                "type": "fully connected",
                "opcode": "add",
                "plastic" : "false",
                "weight config" : {
                    "type" : "flat",
                    "weight" : "0.0",
                },
            })
        gate_index = get_gate_index(from_layer, to_layer, "U")
        connections.append({
            "name": to_layer + "<"+  from_layer + "-input-update",
            "dendrite": to_layer + "<"+  from_layer,
            "from layer": "GATES",
            "to layer": to_layer,
            "type": "subset",
            "subset config" : {
                "from row start" : 0,
                "from row end" : 1,
                "from column start" : gate_index,
                "from column end" : gate_index+1,
                "to row start" : 0,
                "to row end" : 1 if to_layer == "GATES" else 32,
                "to column start" : 0,
                "to column end" : N_GH if to_layer == "GATES" else 32,
            },
            "opcode": "mult_heaviside",
            "plastic" : "false",
            "weight config" : {
                "type" : "flat",
                "weight" : 1
            },
        })

    for layer in LAYERS + DEVICES + ["GATES"]:
        if layer in ["CMPH", "CMPO"]: continue
        connections.append({
            "name": layer + "<"+  layer + "-gain",
            "dendrite": "gain",
            "from layer": layer,
            "to layer": layer,
            "type": "one to one",
            "opcode": "mult",
            "plastic" : "false",
            "weight config" : {
                "type" : "flat",
                "weight" : LAMBDA,
            },
        })

        update_gate_index = get_gate_index(layer, layer, "U")
        connections.append({
            "name": layer + "<"+  "GATES" + "-update-bias",
            "dendrite": "gain-update",
            "from layer": "bias",
            "to layer": layer,
            "type": "fully connected",
            "opcode": "add",
            "plastic" : "false",
            "weight config" : {
                "type" : "flat",
                "weight" : 1
            },
        })
        connections.append({
            "name": layer + "<"+  "GATES" + "-update",
            "dendrite": "gain-update",
            "from layer": "GATES",
            "to layer": layer,
            "type": "subset",
            "subset config" : {
                "from row start" : 0,
                "from row end" : 1,
                "from column start" : update_gate_index,
                "from column end" : update_gate_index+1,
                "to row start" : 0,
                "to row end" : 1 if layer == "GATES" else 32,
                "to column start" : 0,
                "to column end" : N_GH if layer == "GATES" else 32,
            },
            "opcode": "sub_heaviside",
            "plastic" : "false",
            "weight config" : {
                "type" : "flat",
                "weight" : 1
            },
        })

        decay_gate_index = get_gate_index(layer, layer, "C")
        connections.append({
            "name": layer + "<"+  "GATES" + "-decay-bias",
            "dendrite": "gain-decay",
            "from layer": "bias",
            "to layer": layer,
            "type": "fully connected",
            "opcode": "add",
            "plastic" : "false",
            "weight config" : {
                "type" : "flat",
                "weight" : 1
            },
        })
        connections.append({
            "name": layer + "<"+  "GATES" + "-decay",
            "dendrite": "gain-decay",
            "from layer": "GATES",
            "to layer": layer,
            "type": "subset",
            "subset config" : {
                "from row start" : 0,
                "from row end" : 1,
                "from column start" : decay_gate_index,
                "from column end" : decay_gate_index+1,
                "to row start" : 0,
                "to row end" : 1 if layer == "GATES" else 32,
                "to column start" : 0,
                "to column end" : N_GH if layer == "GATES" else 32,
            },
            "opcode": "sub_heaviside",
            "plastic" : "false",
            "weight config" : {
                "type" : "flat",
                "weight" : 1
            },
        })

    cmp_e = 1./(2.*N_LAYER)
    w_cmph = np.arctanh(1. - cmp_e) / (PAD / 2.)**2
    w_cmpo = 2. * np.arctanh(PAD) / (N_LAYER*(1-cmp_e) - (N_LAYER-1))
    b_cmpo = w_cmpo * (N_LAYER*(1 - cmp_e) + (N_LAYER-1)) / 2.

    connections.append({
        "name": "CMPH<bias",
        "from layer": "bias",
        "to layer": "CMPH",
        "type": "fully connected",
        "opcode": "add",
        "plastic" : "false",
        "weight config" : {
            "type" : "flat",
            "weight" : w_cmph
        },
    })

    connections.append({
        "name": "CMPH<CMPA",
        "from layer": "CMPA",
        "to layer": "CMPH",
        "type": "one to one",
        "opcode": "mult",
        "plastic" : "false",
        "weight config" : {
            "type" : "flat",
            "weight" : 1
        },
    })

    connections.append({
        "name": "CMPH<CMPB",
        "from layer": "CMPB",
        "to layer": "CMPH",
        "type": "one to one",
        "opcode": "mult",
        "plastic" : "false",
        "weight config" : {
            "type" : "flat",
            "weight" : 1
        },
    })

    connections.append({
        "name": "CMPO<CMPH",
        "from layer": "CMPH",
        "to layer": "CMPO",
        "type": "fully connected",
        "opcode": "add",
        "plastic" : "false",
        "weight config" : {
            "type" : "flat",
            "weight" : w_cmpo
        },
    })

    connections.append({
        "name": "CMPO<bias",
        "from layer": "bias",
        "to layer": "CMPH",
        "type": "fully connected",
        "opcode": "sub",
        "plastic" : "false",
        "weight config" : {
            "type" : "flat",
            "weight" : b_cmpo
        },
    })

    return structures, connections
