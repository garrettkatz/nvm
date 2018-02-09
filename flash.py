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

def get_gates(p):
    """
    Returns a dict of gate values from a pattern
    """
    g = {}
    for i in range(len(GATE_KEYS)):
        g[GATE_KEYS[i]] = p[i,0]
    return g
        
def initial_pad_gates():
    v = -PAD*np.ones((2*N_GATES, 1)) # start with all gates closed
    v[GATE_INDEX["GATES","GATES","U"],0] = PAD # except gates always updating
    return v

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
V_gos[N_GATES:, 1:] = PAD*np.sign(np.random.randn(N_GATES, GOS-1)) # random hidden
for s in range(GOS):
    for k in GATE_OP_SEQUENCE[s]:
        # print(s,GATE_INDEX[k])
        V_gos[GATE_INDEX[k], s] = PAD

X, Y = [], []

# SET
# 1. if opcode has SET, open (gates,op1) update
# 2. if op1 has <to_layer>, copy op2 to <to_layer>
# 3. return to initial gates
h_set = PAD*np.sign(np.random.randn(N_GATES, 1)) # common first hidden for all SETs
for to_layer in LAYERS + DEVICES:
    steps = 3
    V = np.zeros((V_gos.shape[0] + 3*N_LAYER, GOS + steps))
    V[:V_gos.shape[0],:GOS] = V_gos
    # initialize remainder with closed gates and random hidden before return
    V[:V_gos.shape[0],GOS:] = initial_pad_gates() * np.ones((1,steps))
    V[N_GATES:2*N_GATES,GOS:-1] = PAD*np.sign(np.random.randn(N_GATES, steps-1))
    # 1. if opcode has SET, open (gates,op1) update
    V[V_gos.shape[0]:,[GOS-1]] = PAD*np.sign(np.concatenate((
        TOKENS["SET"], # opcode
        np.zeros((2*N_LAYER,1)), # op1,op2 closed
    ), axis=0))
    V[N_GATES:2*N_GATES,[GOS]] = h_set
    V[GATE_INDEX[("GATES","OPERAND1","U")], GOS] = PAD
    # 2. if op1 has <to_layer>, copy op2 to <to_layer> update
    V[V_gos.shape[0]:,[GOS]] = PAD*np.sign(np.concatenate((
        np.zeros((N_LAYER,1)), # opcode closed
        TOKENS[to_layer],
        np.zeros((N_LAYER,1)), # op2 closed
    ), axis=0))
    V[GATE_INDEX[(to_layer,"OPERAND2","U")], GOS+1] = PAD
    V[GATE_INDEX[(to_layer,to_layer,"C")], GOS+1] = PAD
    # 3. return to initial gates
    V[V_gos.shape[0]:,[GOS+1]] = PAD*np.sign(np.concatenate((
        np.zeros((3*N_LAYER,1)), # all closed
    ), axis=0))

    X.append(V[:,:-1])
    Y.append(V[:2*N_GATES,1:])

# LOAD
# 1. if opcode has LOAD, open (gates,op1), (gates,op2) updates
# 2. if op1 has <to_layer> and op2 has <from_layer>, copy <from_layer> to <to_layer>
# 3. return to initial gates
h_load = PAD*np.sign(np.random.randn(N_GATES, 1)) # common first hidden for all LOADs
for to_layer in LAYERS + DEVICES:
    for from_layer in LAYERS + DEVICES:
        if from_layer in ["OPCODE","OPERAND1","OPERAND2"]: continue
        if to_layer in ["OPCODE","OPERAND1","OPERAND2"]: continue
        steps = 3
        V = np.zeros((V_gos.shape[0] + 3*N_LAYER, GOS + steps))
        V[:V_gos.shape[0],:GOS] = V_gos
        # initialize remainder with closed gates and random hidden before return
        V[:V_gos.shape[0],GOS:] = initial_pad_gates() * np.ones((1,steps))
        V[N_GATES:2*N_GATES,GOS:-1] = PAD*np.sign(np.random.randn(N_GATES, steps-1))
        # 1. if opcode has LOAD, open (gates,op1), (gates,op2) updates
        V[V_gos.shape[0]:,[GOS-1]] = PAD*np.sign(np.concatenate((
            TOKENS["LOAD"], # opcode
            np.zeros((2*N_LAYER,1)), # op1,op2 closed
        ), axis=0))
        V[N_GATES:2*N_GATES,[GOS]] = h_load
        V[GATE_INDEX[("GATES","OPERAND1","U")], GOS] = PAD
        V[GATE_INDEX[("GATES","OPERAND2","U")], GOS] = PAD
        # 2. if op1 has <to_layer> and op2 has <from_layer>, copy <from_layer> to <to_layer>
        V[V_gos.shape[0]:,[GOS]] = PAD*np.sign(np.concatenate((
            np.zeros((N_LAYER,1)), # opcode closed
            TOKENS[to_layer],
            TOKENS[from_layer],
        ), axis=0))
        V[GATE_INDEX[(to_layer,from_layer,"U")], GOS+1] = PAD
        V[GATE_INDEX[(to_layer,to_layer,"C")], GOS+1] = PAD
        # 3. return to initial gates
        V[V_gos.shape[0]:,[GOS+1]] = PAD*np.sign(np.concatenate((
            np.zeros((3*N_LAYER,1)), # all closed
        ), axis=0))
    
        X.append(V[:,:-1])
        Y.append(V[:2*N_GATES,1:])

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
w = flash(X, Y)

WEIGHTS[("GATES","GATES")] = w[:,:2*N_GATES]
WEIGHTS[("GATES","OPCODE")] = w[:,2*N_GATES+0*N_LAYER:2*N_GATES+1*N_LAYER]
WEIGHTS[("GATES","OPERAND1")] = w[:,2*N_GATES+1*N_LAYER:2*N_GATES+2*N_LAYER]
WEIGHTS[("GATES","OPERAND2")] = w[:,2*N_GATES+2*N_LAYER:2*N_GATES+3*N_LAYER]

plt.subplot(1,3,1)
plt.imshow(np.kron((X-X.min())/(X.max()-X.min()),np.ones((1,3))), cmap='gray')
plt.subplot(1,3,2)
plt.imshow(np.kron((Y-Y.min())/(Y.max()-Y.min()),np.ones((1,3))), cmap='gray')
plt.subplot(1,3,3)
plt.imshow((w-w.min())/(w.max()-w.min()), cmap='gray')
plt.show()

raw_input("continue?")
