import numpy as np
from tokens import LAYERS, N_LAYER, PAD

def flash(X, Y, B=None):
    """
    Flash W for Y = np.tanh(W.dot(X) + B)
    """
    if B is None: B = np.zeros((Y.shape[0],1))
    W = np.linalg.lstsq(X.T, (np.arctanh(Y) - B).T, rcond=None)[0].T
    return W

# dict of inter/intra layer weight matrices
WEIGHTS = {}

# map indices in gate layer to keys
GATE_KEYS = []
GATE_INDEX = {}
for layerA in LAYERS:
    for layerB in LAYERS:
        for mode in ["A","L"]: # activate/learn
            GATE_INDEX[(layerA, layerB, mode)] = len(GATE_KEYS)
            GATE_KEYS.append((layerA, layerB, mode))

N_GATES = len(GATE_KEYS)

def get_gates(p):
    """
    Returns a dict of gate values from a pattern
    """
    g = {}
    for i in range(len(GATE_KEYS)):
        g[GATE_KEYS[i]] = p[i,0]
    return g
        
# Gate sequence for loading ops: list open gates at each timestep
GATE_OP_SEQUENCE = [
    [],
    [("MEM","MEM","A"), ("OPCODE","MEM","A")],
    [("MEM","MEM","A"), ("OPERAND1","MEM","A")],
    [("MEM","MEM","A"), ("OPERAND2","MEM","A")],
    []
]

V = -PAD*np.ones((N_GATES, len(GATE_OP_SEQUENCE)))
V = np.concatenate((V, PAD*np.sign(np.random.randn(*V.shape))),axis=0) # hidden
V[N_GATES:,0] = -PAD
for s in range(len(GATE_OP_SEQUENCE)):
    for k in GATE_OP_SEQUENCE[s]:
        V[GATE_INDEX[k], s] = PAD

WEIGHTS[("GATE","GATE")] = flash(V[:,:-1], V[:,1:])

