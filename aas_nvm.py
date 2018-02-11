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
        WEIGHTS[(to_layer,from_layer)] = np.eye(N_LAYER,N_LAYER) * np.arctanh(PAD)/PAD
# ROM
WEIGHTS[("GATES","GATES")] = W_ROM[:,:N_GH]
WEIGHTS[("GATES","OPC")] = W_ROM[:,N_GH+0*N_LAYER:N_GH+1*N_LAYER]
WEIGHTS[("GATES","OP1")] = W_ROM[:,N_GH+1*N_LAYER:N_GH+2*N_LAYER]
WEIGHTS[("GATES","OP2")] = W_ROM[:,N_GH+2*N_LAYER:N_GH+3*N_LAYER]
WEIGHTS[("GATES","OP3")] = W_ROM[:,N_GH+3*N_LAYER:N_GH+4*N_LAYER]


def print_state(activity):
    hr = ["%s:%s"%(k,get_token(activity[k])) for k in [
        "OPC","OP1","OP2","OP3","MEM","REG1","REG2","REG3","FEF","TC"
    ]]
    open_gates = get_open_gates(activity["GATES"])
    print(" ".join(hr))
    print("open gates: " + str(tuple(open_gates)))

def tick(activity, weights):

    # NVM tick
    activity_new = {k: np.zeros(v.shape) for (k,v) in activity.items()}
    for (to_layer, from_layer) in weights:
        u = get_gates(activity["GATES"])[(to_layer, from_layer, "U")]
        u = float(u > 0)
        w = weights[(to_layer, from_layer)]
        w = u*w
        if to_layer == from_layer:
            c = get_gates(activity["GATES"])[(to_layer, from_layer, "C")]
            c = float(c > 0)
            w += (1-u)*(1-c)*np.eye(*w.shape) * np.arctanh(PAD)/PAD
        activity_new[to_layer] += w.dot(activity[from_layer])
    
    # handle compare specially, never gated
    cmp_e = 1./(2.*N_LAYER)
    w_cmph = np.arctanh(1. - cmp_e) / (PAD / 2.)**2
    w_cmpo = 2. * np.arctanh(PAD) / (N_LAYER*(1-cmp_e) - (N_LAYER-1))
    b_cmpo = w_cmpo * (N_LAYER*(1 - cmp_e) + (N_LAYER-1)) / 2.
    activity_new["CMPH"] = w_cmph * activity["CMPA"] * activity["CMPB"]
    activity_new["CMPO"] = np.ones((N_LAYER,1)) * (w_cmpo * activity["CMPH"].sum() - b_cmpo)
    
    for layer in activity_new:
        activity_new[layer] = np.tanh(activity_new[layer])

    return activity_new
