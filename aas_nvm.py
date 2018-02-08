import numpy as np
from tokens import get_token, N_LAYER, LAYERS, DEVICES
from flash import get_gates, N_GATES, PAD

def print_state(ACTIVITY):
    hr = ["%s:%s"%(k,get_token(ACTIVITY[k])) for k in [
        "OPCODE","OPERAND1","OPERAND2","MEM1","FEF","TC"
    ]]
    all_gates = get_gates(ACTIVITY["GATES"])
    open_gates = [k for k in all_gates if all_gates[k] > 0]
    print(" ".join(hr))
    print("open gates: " + str(tuple(open_gates)))

def tick(ACTIVITY, WEIGHTS):

    # NVM tick
    ACTIVITY_NEW = {k: np.zeros(v.shape) for (k,v) in ACTIVITY.items()}
    for (to_layer, from_layer) in WEIGHTS:
        u = get_gates(ACTIVITY["GATES"])[(to_layer, from_layer, "U")]
        c = get_gates(ACTIVITY["GATES"])[(to_layer, from_layer, "C")]
        u, c = float(u > 0), float(c > 0)
        w = WEIGHTS[(to_layer, from_layer)]
        w = u*w
        if to_layer == from_layer:
            w += (1-u)*(1-c)*np.eye(*w.shape) * np.arctanh(PAD)/PAD
        ACTIVITY_NEW[to_layer] += w.dot(ACTIVITY[from_layer])
    
    for layer in ACTIVITY_NEW:
        ACTIVITY_NEW[layer] = np.tanh(ACTIVITY_NEW[layer])

    return ACTIVITY_NEW
