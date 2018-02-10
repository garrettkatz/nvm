import numpy as np
from tokens import get_token, N_LAYER, LAYERS, DEVICES
from gates import get_gates, PAD

def print_state(activity):
    hr = ["%s:%s"%(k,get_token(activity[k])) for k in [
        "OPCODE","OPERAND1","OPERAND2","MEM1","FEF","TC"
    ]]
    all_gates = get_gates(activity["GATES"])
    open_gates = [k for k in all_gates if all_gates[k] > 0]
    print(" ".join(hr))
    print("open gates: " + str(tuple(open_gates)))

def tick(activity, WEIGHTS):

    # NVM tick
    activity_new = {k: np.zeros(v.shape) for (k,v) in activity.items()}
    for (to_layer, from_layer) in WEIGHTS:
        u = get_gates(activity["GATES"])[(to_layer, from_layer, "U")]
        c = get_gates(activity["GATES"])[(to_layer, from_layer, "C")]
        u, c = float(u > 0), float(c > 0)
        w = WEIGHTS[(to_layer, from_layer)]
        w = u*w
        if to_layer == from_layer:
            w += (1-u)*(1-c)*np.eye(*w.shape) * np.arctanh(PAD)/PAD
        activity_new[to_layer] += w.dot(activity[from_layer])
    
    for layer in activity_new:
        activity_new[layer] = np.tanh(activity_new[layer])

    return activity_new
