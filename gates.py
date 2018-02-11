import numpy as np
import matplotlib.pyplot as plt
from tokens import LAYERS, DEVICES, N_LAYER, TOKENS, get_token

PAD = 0.9

np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .0f'%x})

# map indices in gate layer to keys
# gate modes:
# default is for activity pattern to remain fixed
# if "clear" gate is open within region, activity pattern vanishes (for overwrite)
# if "update" gate is open, weight matrix applies
    # within region, activity pattern transitions
    # between regions, signals are propogated (needs clear destination for copy)
GATE_KEYS = []
GATE_INDEX = {}
# for to_layer in LAYERS + DEVICES + ["GATES"]:
#     for from_layer in LAYERS + DEVICES + ["GATES"]:
#         # for mode in ["C","U","L"]: # clear/update/learn
#         for mode in ["C","U"]:
#             GATE_INDEX[(to_layer, from_layer, mode)] = len(GATE_KEYS)
#             GATE_KEYS.append((to_layer, from_layer, mode))
for to_layer in LAYERS + DEVICES + ["GATES"]:
    for from_layer in LAYERS + DEVICES + ["GATES"]:
        # for mode in ["U","L"]: # clear/update/learn
        for mode in ["U"]:
            GATE_INDEX[(to_layer, from_layer, mode)] = len(GATE_KEYS)
            GATE_KEYS.append((to_layer, from_layer, mode))
    GATE_INDEX[(to_layer, to_layer, "C")] = len(GATE_KEYS)
    GATE_KEYS.append((to_layer, to_layer, "C"))
    

N_GATES = len(GATE_KEYS)
N_HGATES = 256
N_GH = N_GATES + N_HGATES

def get_gates(p):
    """Returns a dict of gate values from a pattern"""
    g = {}
    for i in range(len(GATE_KEYS)):
        g[GATE_KEYS[i]] = p[i,0]
    return g
    
def get_open_gates(p):
    """Get human-readable list of open gates in pattern"""
    all_gates = get_gates(p)
    return [k for k in all_gates if all_gates[k] > 0]

def default_gates():
    """All closed except the (gate,gate) update"""
    gates = -PAD*np.ones((N_GH,1))
    gates[GATE_INDEX[("GATES","GATES","U")],0] = PAD
    return gates

