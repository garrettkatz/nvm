import numpy as np
from tokens import get_token, N_LAYER, LAYERS, DEVICES
from gates import get_gates, get_open_gates, PAD, N_GH
from flash_rom import W_ROM

#### FINALIZE WEIGHTS ###

# dict of inter/intra layer weight matrices
WEIGHTS = {}
# relays
for to_layer in LAYERS + DEVICES:
    for from_layer in LAYERS + DEVICES:
        WEIGHTS[(to_layer,from_layer)] = "relay" #np.eye(N_LAYER,N_LAYER) * np.arctanh(PAD)/PAD
# RAM (will be learned)
WEIGHTS[("MEM","MEM")] = "none"
WEIGHTS[("MEM","MEMH")] = np.zeros((N_LAYER, N_LAYER))
WEIGHTS[("MEMH","MEM")] = "none"
WEIGHTS[("MEMH","MEMH")] = np.zeros((N_LAYER, N_LAYER))
# ROM
WEIGHTS[("GATES","GATES")] = W_ROM[:,:N_GH]
WEIGHTS[("GATES","OPC")] = W_ROM[:,N_GH+0*N_LAYER:N_GH+1*N_LAYER]
WEIGHTS[("GATES","OP1")] = W_ROM[:,N_GH+1*N_LAYER:N_GH+2*N_LAYER]
WEIGHTS[("GATES","OP2")] = W_ROM[:,N_GH+2*N_LAYER:N_GH+3*N_LAYER]
WEIGHTS[("GATES","OP3")] = W_ROM[:,N_GH+3*N_LAYER:N_GH+4*N_LAYER]

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
            wv = u * np.arctanh(PAD)/PAD * activity[from_layer]
        else:
            wv = u * w.dot(activity[from_layer])
            # # temp memory test
            # if to_layer[:3] == "MEM" and from_layer == "MEMH":
            #     wv = np.arctanh(PAD)*np.sign(wv)
        if to_layer == from_layer:
            c = current_gates[(to_layer, from_layer, "C")]
            c = float(c > 0)
            wv += (1-u)*(1-c)*np.arctanh(PAD)/PAD * activity[from_layer]
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
