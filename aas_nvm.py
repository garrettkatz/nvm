import numpy as np
import scipy.sparse as sp
from tokens import get_token, N_LAYER, LAYERS, USER_LAYERS, DEVICES
from gates import get_gates, get_open_gates, PAD, N_GH
from flash_rom import W_ROM

#### FINALIZE WEIGHTS ###

# dict of inter/intra layer weight matrices
WEIGHTS = {}

# copy connections
USR = USER_LAYERS + DEVICES
relays = []
relays += [(opx, "MEM") for opx in ["OPC","OP1","OP2","OP3"]] # MEM to ops (clock)
relays += [(usr, "OP2") for usr in USR] # op2 to user (set)
relays += [(usr1, usr2) for usr1 in USR for usr2 in USR] # user to each other (mov)
relays += [("CMP"+ab, usr) for ab in "AB" for usr in USR] # user to cmpa/b (cmp)
relays += [(usr, "CMPO") for usr in USR] # cmpo to user (cmp)
relays += [("OP1", usr) for usr in USR] # user to op1 (jmp)
relays += [("MEM"+h, "OP"+o) for h in ["","H"] for o in "23"] # op2 to MEM, op3 to MEMH (jmp)

# relays
for (to_layer, from_layer) in relays:
    WEIGHTS[(to_layer,from_layer)] = sp.eye(N_LAYER) * np.arctanh(PAD)/PAD

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
        "OPC","OP1","OP2","OP3","REG1","REG2","REG3","FEF","TC"
    ]]
    s = " ".join(hr)
    # open_gates = get_open_gates(activity["GATES"])
    # s = s + "%nopen gates: " + str(tuple(open_gates)))
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
        w = u*w
        if to_layer == from_layer:
            c = current_gates[(to_layer, from_layer, "C")]
            c = float(c > 0)
            w += (1-u)*(1-c)*sp.eye(*w.shape) * np.arctanh(PAD)/PAD
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
        # inject noise
        flip = (np.random.rand(activity_new[layer].shape[0]) < FLIP_NOISE)
        activity_new[layer][flip,:] = -activity_new[layer][flip,:]
        activity_new[layer] += np.random.randn(*activity_new[layer].shape)*CTS_NOISE

    return activity_new

def hebb_update(x, y, w):
    N = x.shape[0]
    w += np.arctanh(y[:,[0]]) * x[:,[0]].T #/(N*PAD**2)
    return w
