import numpy as np
from sys import exit
from tokens import get_token, N_LAYER, N_LAYER_DIM, LAYERS, DEVICES, TOKENS
from gates import get_gates, get_open_gates, get_gate_index, PAD, LAMBDA, N_GATES, N_HGATES, N_GH
from flash_rom import W_ROM
from syngen import ConnectionFactory

# Comparison values
# activity_new["CMPH"] = W_CMPH * activity["CMPA"] * activity["CMPB"]
# activity_new["CMPO"] = np.ones((N_LAYER,1)) * (W_CMPO * activity["CMPH"].sum() - B_CMPO)
CMP_E = 1./(2.*N_LAYER)
W_CMPH = np.arctanh(1. - CMP_E) / (PAD / 2.)**2
W_CMPO = 2. * np.arctanh(PAD) / (N_LAYER*(1-CMP_E) - (N_LAYER-1))
B_CMPO = W_CMPO * (N_LAYER*(1 - CMP_E) + (N_LAYER-1)) / 2.


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
        exit(0)
    
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
    activity_new["CMPH"] = W_CMPH * activity["CMPA"] * activity["CMPB"]
    activity_new["CMPO"] = np.ones((N_LAYER,1)) * (W_CMPO * activity["CMPH"].sum() - B_CMPO)
    
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
    comp_external_layers = ["CMPA", "CMPB"]
    comp_internal_layers = ["CMPH", "CMPO"]
    other_layers = [
        l for l in LAYERS + DEVICES + ["GATES"]
            if l not in comp_internal_layers + comp_external_layers]

    layer_configs = []
    for layer in other_layers + comp_external_layers:
        dendrites = []
        # Add dendrites for other layers
        for (to_layer,from_layer),w in weights.iteritems():
            if to_layer == layer and \
                (type(w) is not str or (type(w) is str and w != "none")):
                dendrites.append({"name" : layer + "<" + from_layer})

        # Add dendrites for update/decay gain
        dendrites.append(
            {
                "name" : "gain",
                "children" : [
                    {
                        "name" : "gain-update",
                        "opcode" : "add",
                        "init" : 1.0 # bias
                    },
                    {
                        "name" : "gain-decay",
                        "opcode" : "mult",
                        "init" : 1.0 # bias
                    }
                ]
            }
        )

        # Build layer config
        layer_configs.append({
            "name" : layer,
            "neural model" : "nvm",
            "dendrites" : dendrites,
            "rows" : 1 if layer == "GATES" else N_LAYER_DIM,
            "columns" : N_GH if layer == "GATES" else N_LAYER_DIM
        })

    layer_configs.append({
        "name" : "CMPH",
        "neural model" : "nvm",
        "rows" : N_LAYER_DIM,
        "columns" : N_LAYER_DIM,
        "noise config" : {
            "type" : "flat",
            "val" : W_CMPH  # bias
        }
    })
    layer_configs.append({
        "name" : "CMPO",
        "neural model" : "nvm",
        "rows" : N_LAYER_DIM,
        "columns" : N_LAYER_DIM,
        "noise config" : {
            "type" : "flat",
            "val" : -B_CMPO  # bias
        }
    })

    structures = [{"name" : "nvm",
                   "type" : "parallel",
                   "layers": layer_configs}]

    # Parameters shared by all connections
    defaults = { "plastic" : "false" }

    # Builds a name for a connection or dendrite
    def build_name(from_layer, to_layer, suffix=""):
        return to_layer + "<" + from_layer + suffix

    # Build relay (one to one) connection
    def build_relay(from_layer, to_layer, props):
        props["name"] = build_name(from_layer, to_layer, "input")
        props["from layer"] = from_layer
        props["to layer"] = to_layer
        props["dendrite"] = build_name(from_layer, to_layer)
        props["type"] = "one to one"
        props["opcode"] = "add"
        props["weight config"] = {
            "type" : "flat",
            "weight" : LAMBDA
        }

    # Build fully connected connection
    def build_full(from_layer, to_layer, props):
        props["name"] = build_name(from_layer, to_layer, "-input")
        props["from layer"] = from_layer
        props["to layer"] = to_layer
        props["dendrite"] = build_name(from_layer, to_layer)
        props["type"] = "fully connected"
        props["opcode"] = "add"
        props["weight config"] = {
            "type" : "flat",
            "weight" : 0.0  # This will get overwritten when the ROM is flashed
        }

    # Build gate for another connection
    def build_update(from_layer, to_layer, props):
        props["name"] = build_name(from_layer, to_layer, "-input-update")
        props["from layer"] = "GATES"
        props["to layer"] = to_layer
        props["dendrite"] = build_name(from_layer, to_layer)
        props["opcode"] = "mult_heaviside"
        props["weight config"] = {
            "type" : "flat",
            "weight" : 1
        }
        props["type"] = "subset"

        gate_index = get_gate_index(to_layer, from_layer, "U")
        props["subset config"] = {
            "from row start" : 0,
            "from row end" : 1,
            "from column start" : gate_index,
            "from column end" : gate_index+1,
            "to row start" : 0,
            "to row end" : 1 if to_layer == "GATES" else N_LAYER_DIM,
            "to column start" : 0,
            "to column end" : N_GH if to_layer == "GATES" else N_LAYER_DIM,
        }

    # Builds a multiplicative gain connection (from_layer = to_layer)
    def build_gain(from_layer, to_layer, props):
        props["name"] = build_name(from_layer, to_layer, "-gain")
        props["from layer"] = from_layer
        props["to layer"] = to_layer
        props["dendrite"] = "gain"
        props["type"] = "one to one"
        props["opcode"] = "mult"
        props["weight config"] = {
            "type" : "flat",
            "weight" : LAMBDA
        }

    # Builds update gate of gain (from_layer = GATES)
    def build_gain_update(from_layer, to_layer, props):
        props["name"] = build_name(from_layer, to_layer, "-update")
        props["from layer"] = from_layer
        props["to layer"] = to_layer
        props["dendrite"] = "gain-update"
        props["opcode"] = "sub_heaviside"  # (1 - hs(u)), dendrite has bias
        props["weight config"] = {
            "type" : "flat",
            "weight" : 1
        }
        props["type"] = "subset"

        update_gate_index = get_gate_index(to_layer, to_layer, "U")
        props["subset config"] = {
            "from row start" : 0,
            "from row end" : 1,
            "from column start" : update_gate_index,
            "from column end" : update_gate_index+1,
            "to row start" : 0,
            "to row end" : 1 if to_layer == "GATES" else N_LAYER_DIM,
            "to column start" : 0,
            "to column end" : N_GH if to_layer == "GATES" else N_LAYER_DIM,
        }

    # Builds decay gate of gain (from_layer = GATES)
    def build_gain_decay(from_layer, to_layer, props):
        props["name"] = build_name(from_layer, to_layer,"-decay")
        props["from layer"] = from_layer
        props["to layer"] = to_layer
        props["dendrite"] = "gain-decay"
        props["opcode"] = "sub_heaviside"  # (1 - hs(d)), dendrite has bias
        props["weight config"] = {
            "type" : "flat",
            "weight" : 1
        }
        props["type"] = "subset"

        decay_gate_index = get_gate_index(to_layer, to_layer, "C")
        props["subset config"] = {
            "from row start" : 0,
            "from row end" : 1,
            "from column start" : decay_gate_index,
            "from column end" : decay_gate_index+1,
            "to row start" : 0,
            "to row end" : 1 if to_layer == "GATES" else N_LAYER_DIM,
            "to column start" : 0,
            "to column end" : N_GH if to_layer == "GATES" else N_LAYER_DIM,
        }

    # Create factories
    relay_factory = ConnectionFactory(defaults, build_relay)
    full_factory = ConnectionFactory(defaults, build_full)
    update_factory = ConnectionFactory(defaults, build_update)
    gain_factory = ConnectionFactory(defaults, build_gain)
    gain_update_factory = ConnectionFactory(defaults, build_gain_update)
    gain_decay_factory = ConnectionFactory(defaults, build_gain_decay)

    connections = []

    # Build standard connections and their gates
    for (to_layer, from_layer),w in weights.iteritems():
        if to_layer in ["CMPH", "CMPO"] \
            or from_layer in ["CMPH", "CMPA", "CMPB"] \
            or (type(w) == str and w == "none"):
            continue

        if type(w) == str and w == "relay":
            # wv = u * LAMBDA * activity[from_layer]
            connections.append(relay_factory.build(from_layer, to_layer))
        else:
            # wv = u * w.dot(activity[from_layer])
            connections.append(full_factory.build(from_layer, to_layer))

        connections.append(update_factory.build(from_layer, to_layer))

    # Add gain connections
    for layer in other_layers + comp_external_layers:
        connections.append(gain_factory.build(layer, layer))
        connections.append(gain_update_factory.build("GATES", layer))
        connections.append(gain_decay_factory.build("GATES", layer))

    # Build comparison connections
    for from_layer in comp_external_layers:
        connections.append({
            "name": build_name(from_layer, "CMPH"),
            "from layer": from_layer,
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
            "weight" : W_CMPO
        },
    })

    return structures, connections
