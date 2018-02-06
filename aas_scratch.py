import numpy as np
from tokens import N_LAYER, LAYERS, DEVICES, PAD, TOKENS, PATTERNS, get_token
from flash import flash, WEIGHTS, N_GATES, get_gates

np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .3f'%x})

program = [
    "SET", "FEF", "CENTER",
    "LOAD", "COMPARE1", "TC",
    "LOAD", "COMPARE2", "LEFT",
    "LOAD", "REGISTER1", "COMPARE3",
    "IF", "REGISTER1", "LLEFT",
    "LOAD", "COMPARE2", "RIGHT",
    "LOAD", "REGISTER1", "COMPARE3",
    "IF", "REGISTER1", "LRIGHT",
    "GOTO", "LCENTER", "NULL",
    "SET", "FEF", "LEFT",
    "GOTO", "LCENTER", "NULL",
    "SET", "FEF", "RIGHT",
    "GOTO", "LCENTER", "NULL",
    "RET",
]
labels = {
    "LCENTER": 0,
    "LLEFT": 27,
    "LRIGHT": 33,
}

V = PAD*np.concatenate(tuple(TOKENS[t] for t in program), axis=1) # program
V = np.concatenate((V, PAD*np.sign(np.random.randn(*V.shape))),axis=0) # add hidden

WEIGHTS[("MEM","MEM")] = flash(V[:,:-1], V[:,1:])

ACTIVITY = {k: np.zeros((2*N_LAYER,1)) for k in LAYERS}
for k in DEVICES:
    ACTIVITY[k] = np.zeros((N_LAYER,1))
ACTIVITY["GATE"] = -PAD*np.ones((2*N_GATES,1))
ACTIVITY["MEM"] = V[:,[0]]

for t in range(10):
    token = get_token(ACTIVITY["MEM"][:N_LAYER,:])
    print("%d: "%t + " ".join(["%s:%s"%(k,get_token(ACTIVITY[k][:N_LAYER,:])) for k in ["MEM","FEF"]]))
    if token == "RET": break
    all_gates = get_gates(ACTIVITY["GATE"][:N_GATES,:])
    open_gates = [k for k in all_gates if all_gates[k] > 0]
    print("open gates: " + str(tuple(open_gates)))
    
    # NVM tick
    ACTIVITY_NEW = {}
    
    g = get_gates(ACTIVITY["GATE"])[("MEM","MEM","A")]
    w = WEIGHTS[("MEM","MEM")]
    w = (g+1)/2*w + (1-(g+1)/2)*2*np.eye(*w.shape)
    ACTIVITY_NEW["MEM"] = w.dot(ACTIVITY["MEM"])

    w = WEIGHTS[("GATE","GATE")]
    ACTIVITY_NEW["GATE"] = w.dot(ACTIVITY["GATE"])
    
    for k in ["MEM","GATE"]:
        ACTIVITY_NEW[k] = np.tanh(ACTIVITY_NEW[k])

    for k in DEVICES:
        ACTIVITY_NEW[k] = ACTIVITY[k]

    ACTIVITY = ACTIVITY_NEW

    # ACTIVITY["GATE"][0,0] = 1.
