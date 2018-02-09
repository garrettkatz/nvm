import numpy as np
import matplotlib.pyplot as plt
from tokens import LAYERS, DEVICES, N_LAYER, TOKENS, get_token

PAD = 0.9

np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .0f'%x})

def flash(X, Y, B=None):
    """
    Flash W for Y = np.tanh(W.dot(X) + B)
    """
    if B is None: B = np.zeros((Y.shape[0],1))
    W = np.linalg.lstsq(X.T, (np.arctanh(Y) - B).T, rcond=None)[0].T
    print("Flash residual: %f"%np.fabs(Y - np.tanh(W.dot(X)+B)).max())
    return W

# dict of inter/intra layer weight matrices
WEIGHTS = {}
        
# relays
for to_layer in LAYERS + DEVICES:
    for from_layer in LAYERS + DEVICES:
        WEIGHTS[(to_layer,from_layer)] = np.eye(N_LAYER,N_LAYER) * np.arctanh(PAD)/PAD

# map indices in gate layer to keys
# gate modes:
# a region->region connection:
# if update is open, normal signals are propagated
# if clear is open, no signals are propogated (destination goes to center)
# if neither are open, one-to-one signals are propogated (destination copies)
# update gate open: the region will transition to next attractor
# clear gate open: the region will clear activity 
GATE_KEYS = []
GATE_INDEX = {}
for to_layer in LAYERS + DEVICES + ["GATES"]:
    for from_layer in LAYERS + DEVICES + ["GATES"]:
        # for mode in ["C","U","L"]: # clear/update/learn
        for mode in ["C","U"]: # clear/update/learn
            GATE_INDEX[(to_layer, from_layer, mode)] = len(GATE_KEYS)
            GATE_KEYS.append((to_layer, from_layer, mode))

N_GATES = len(GATE_KEYS)
N_HGATES = 256
N_GH = N_GATES + N_HGATES

def get_gates(p):
    """
    Returns a dict of gate values from a pattern
    """
    g = {}
    for i in range(len(GATE_KEYS)):
        g[GATE_KEYS[i]] = p[i,0]
    return g
        
def initial_pad_gates():
    v = -PAD*np.ones((N_GH, 1)) # start with all gates closed
    v[GATE_INDEX["GATES","GATES","U"],0] = PAD # except gates always updating
    return v

def make_pad_op_gates(gates, hgates=None, opc=None, op1=None, op2=None):
    # init
    v = np.concatenate((initial_pad_gates(), np.zeros((3*N_LAYER,1))),axis=0)
    # set gates
    for k in gates: v[GATE_INDEX[k],0] = PAD
    # set hidden
    if hgates is None: hgates = np.random.randn(N_HGATES,1)
    v[N_GATES:N_GH,[0]] = hgates
    # set op
    if opc is not None: v[N_GH+0*N_LAYER:N_GH+1*N_LAYER,[0]] = TOKENS[opc]
    if op1 is not None: v[N_GH+1*N_LAYER:N_GH+2*N_LAYER,[0]] = TOKENS[op1]
    if op2 is not None: v[N_GH+2*N_LAYER:N_GH+3*N_LAYER,[0]] = TOKENS[op2]
    return PAD*np.sign(v)

# Gate sequence for loading ops: list open gates at each timestep
GATE_OP_SEQUENCE = [
    [],
    [("MEM1","MEM1","U"), ("MEM1","MEM2","U"),("MEM2","MEM1","U"),("MEM2","MEM2","U"), ("OPCODE","MEM1","U"),("OPCODE","OPCODE","C")],
    [("MEM1","MEM1","U"), ("MEM1","MEM2","U"),("MEM2","MEM1","U"),("MEM2","MEM2","U"), ("OPERAND1","MEM1","U"),("OPERAND1","OPERAND1","C")],
    [("MEM1","MEM1","U"), ("MEM1","MEM2","U"),("MEM2","MEM1","U"),("MEM2","MEM2","U"), ("OPERAND2","MEM1","U"),("OPERAND2","OPERAND2","C")],
    [("GATES","OPCODE","U")],
]

GOS = len(GATE_OP_SEQUENCE)
V_gos = initial_pad_gates() * np.ones((1, GOS))
V_gos[N_GATES:, 1:] = PAD*np.sign(np.random.randn(N_HGATES, GOS-1)) # rand hidden
for s in range(GOS):
    for k in GATE_OP_SEQUENCE[s]:
        V_gos[GATE_INDEX[k], s] = PAD

X, Y = [], []

# initial op sequence
V_gos = np.concatenate((V_gos, np.zeros((3*N_LAYER, V_gos.shape[1]))),axis=0)
# X.append(V_gos[:,:-1])
# Y.append(V_gos[:N_GH,1:])

# initial, ready states before/after op load
V_initial = V_gos[:,[0]].copy()
V_ready = V_gos[:,[-1]].copy()

############# SET
# 1. if opcode has SET, open (gates,op1) update
# 2. if op1 has <to>, clear+update (<to>,op2)
# 3. return to initial gates

for doset in range(1):

    to_layers = DEVICES
    
    # common first step for all sets
    V_set0 = V_ready.copy()
    V_set0[N_GH:N_GH+N_LAYER,:] = TOKENS["SET"]

    h_set1 = np.random.randn(N_HGATES,1)
    # steps 2 and 3 for each to_layer
    V_set1, V_set2 = [], []
    for to_layer in to_layers:
        # 1. if opcode has SET, open (gates,op1) update
        V_set1.append(make_pad_op_gates(
            [("GATES","OPCODE","U"),("GATES","OPERAND1","U")],
            hgates=h_set1,opc="SET",op1=to_layer))
        # 2. if op1 has <to>, clear+update (<to>,op2)
        V_set2.append(make_pad_op_gates(
            [("GATES","OPCODE","U"),("GATES","OPERAND1","U"),
            (to_layer,"OPERAND2","U"),(to_layer, to_layer, "C")],
            opc="SET",op1=to_layer))
    V_set1 = np.concatenate(V_set1,axis=1)
    V_set2 = np.concatenate(V_set2,axis=1)
    # 3. return to initial gates
    V_set3 = V_initial.copy() * np.ones(V_set2.shape)
    
    # set up training data
    X.append(V_set0)
    Y.append(V_set1[:N_GH,[0]])
    X.append(V_set1)
    Y.append(V_set2[:N_GH,:])
    X.append(V_set2)
    Y.append(V_set3[:N_GH,:])

############# LOAD
# 0. (gates,opc) open, h_0, opc = LOAD
# 1. (gates,op1) and (gates,op2) open, h1, op1-op2 have to-from
# 2. h_ to,from (using coefficients A of column space)
# 3. (to, from) open
# 4. return to initial

for doload in range(1):

    # to_layers = LAYERS+DEVICES
    # from_layers = LAYERS+DEVICES
    to_layers = ["FEF"]
    from_layers = DEVICES

    # 0. (gates,opc) open, h_0, opc = LOAD
    V_load0 = V_ready.copy()
    V_load0[N_GH:N_GH+N_LAYER,:] = TOKENS["LOAD"]

    # Set up steps
    h_load1 = np.random.randn(N_HGATES, 1)
    V_load1, V_load2, V_load3 = [], [], []
    for to_layer in to_layers:
        for from_layer in from_layers:
            # 1. (gates,op1) and (gates,op2) open, h1, op1-op2 have to-from
            V_load1.append(make_pad_op_gates(
                [("GATES","OPCODE","U"),("GATES","OPERAND1","U"),
                ("GATES","OPERAND2","U")],
                hgates=h_load1,opc="LOAD",op1=to_layer,op2=from_layer))
            # 2. h_ to,from
            V_load2.append(make_pad_op_gates([
                ("GATES","OPCODE","U"),("GATES","OPERAND1","U"),
                ("GATES","OPERAND2","U")],
                opc="LOAD",op1=to_layer,op2=from_layer))
            # 3. (to, from) open
            V_load3.append(make_pad_op_gates([
                ("GATES","OPCODE","U"),("GATES","OPERAND1","U"),
                ("GATES","OPERAND2","U"),
                (to_layer,to_layer,"C"),(to_layer,from_layer,"U")],
                opc="LOAD",op1=to_layer,op2=from_layer))
    V_load1 = np.concatenate(V_load1,axis=1)
    V_load2 = np.concatenate(V_load2,axis=1)
    V_load3 = np.concatenate(V_load3,axis=1)
    # 4. return to initial
    V_load4 = V_initial.copy() * np.ones(V_load3.shape)

    # arrange h_ to,from using coefficients (A) of column space
    _, s, vh = np.linalg.svd(V_load1, full_matrices=False)
    A = vh[s > s.max() * max(V_load1.shape) * np.finfo(s.dtype).eps, :]
    V_load2[N_GATES:N_GH,:] = np.tanh(np.random.randn(N_HGATES, A.shape[0]).dot(A))

    # set up training data
    X.append(V_load0)
    Y.append(V_load1[:N_GH,[0]])
    X.append(V_load1)
    Y.append(V_load2[:N_GH,:])
    X.append(V_load2)
    Y.append(V_load3[:N_GH,:])
    X.append(V_load3)
    Y.append(V_load4[:N_GH,:])


# check for non-determinism
print('nondet...')
for s1 in range(len(X)):
    break
    for s2 in range(len(X)):
        for i1 in range(X[s1].shape[1]):
            for i2 in range(X[s2].shape[1]):
                if (np.sign(X[s1][:,i1]) == np.sign(X[s2][:,i2])).all() and (np.sign(Y[s1][:,i1]) != np.sign(Y[s2][:,i2])).any():
                    print("non-deterministic!")
                    print((s1,s2,i1,i2))
                    raw_input("continue?!?")


X = np.concatenate(X, axis=1)
Y = np.concatenate(Y, axis=1)

print('X,preY,Y shapes & ranks:')
print(X.shape)
print(np.linalg.matrix_rank(X))
print(Y.shape)
print(np.linalg.matrix_rank(np.arctanh(Y)))
print(Y.shape)
print(np.linalg.matrix_rank(Y))

print("%d gates, %d hgates, %d layers/devices, %d layer size"%(
    N_GATES, N_HGATES, len(LAYERS+DEVICES), N_LAYER
))


w = flash(X, Y)

WEIGHTS[("GATES","GATES")] = w[:,:N_GH]
WEIGHTS[("GATES","OPCODE")] = w[:,N_GH+0*N_LAYER:N_GH+1*N_LAYER]
WEIGHTS[("GATES","OPERAND1")] = w[:,N_GH+1*N_LAYER:N_GH+2*N_LAYER]
WEIGHTS[("GATES","OPERAND2")] = w[:,N_GH+2*N_LAYER:N_GH+3*N_LAYER]

plt.subplot(1,3,1)
plt.imshow(np.kron((X-X.min())/(X.max()-X.min()),np.ones((1,3))), cmap='gray')
plt.subplot(1,3,2)
plt.imshow(np.kron((Y-Y.min())/(Y.max()-Y.min()),np.ones((1,3))), cmap='gray')
plt.subplot(1,3,3)
plt.imshow((w-w.min())/(w.max()-w.min()), cmap='gray')
plt.show()

# raw_input("continue?")
