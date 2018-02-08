import numpy as np
import matplotlib.pyplot as plt
from tokens import N_LAYER, LAYERS, DEVICES, TOKENS, PATTERNS, get_token
from flash import flash, WEIGHTS, N_GATES, PAD, get_gates, initial_pad_gates
from aas_nvm import tick, print_state

np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

program = [
    "SET", "FEF", "CENTER",
    "LOAD", "TC", "FEF",
    # "LOAD", "COMPARE1", "TC",
    # "LOAD", "COMPARE2", "LEFT",
    # "LOAD", "REGISTER1", "COMPARE3",
    # "IF", "REGISTER1", "LLEFT",
    # "LOAD", "COMPARE2", "RIGHT",
    # "LOAD", "REGISTER1", "COMPARE3",
    # "IF", "REGISTER1", "LRIGHT",
    # "GOTO", "LCENTER", "NULL",
    # "SET", "FEF", "LEFT",
    # "GOTO", "LCENTER", "NULL",
    # "SET", "FEF", "RIGHT",
    # "GOTO", "LCENTER", "NULL",
    "RET",
]
labels = {
    "LCENTER": 0,
    "LLEFT": 27,
    "LRIGHT": 33,
}

V = PAD*np.sign(np.concatenate(tuple(TOKENS[t] for t in program), axis=1)) # program
V = np.concatenate((V, PAD*np.sign(np.random.randn(*V.shape))),axis=0) # add hidden

w = flash(V[:,:-1], V[:,1:])
WEIGHTS[("MEM1","MEM1")] = w[:N_LAYER,:N_LAYER]
WEIGHTS[("MEM1","MEM2")] = w[:N_LAYER,N_LAYER:]
WEIGHTS[("MEM2","MEM1")] = w[N_LAYER:,:N_LAYER]
WEIGHTS[("MEM2","MEM2")] = w[N_LAYER:,N_LAYER:]

ACTIVITY = {k: -PAD*np.ones((N_LAYER,1)) for k in LAYERS}
for k in DEVICES:
    ACTIVITY[k] = -PAD*np.ones((N_LAYER,1))
ACTIVITY["GATES"] = initial_pad_gates()
ACTIVITY["MEM1"] = V[:N_LAYER,[0]]
ACTIVITY["MEM2"] = V[N_LAYER:,[0]]

HISTORY = [ACTIVITY]
for t in range(20):
    print("tick %d:"%t)
    print_state(ACTIVITY)
    if get_token(ACTIVITY["OPCODE"]) == "RET": break
    
    ACTIVITY = tick(ACTIVITY, WEIGHTS)
    HISTORY.append(ACTIVITY)

A = np.zeros((2*N_GATES + 5*N_LAYER,len(HISTORY)))
for h in range(len(HISTORY)):
    A[:,[h]] = np.concatenate((
        HISTORY[h]["GATES"],
        HISTORY[h]["OPCODE"],
        HISTORY[h]["OPERAND1"],
        HISTORY[h]["OPERAND2"],
        HISTORY[h]["FEF"],
        HISTORY[h]["TC"],
    ),axis=0)

print(np.fabs(A).max(axis=0))

plt.imshow(np.kron((A-A.min())/(A.max()-A.min()),np.ones((1,20))), cmap='gray')
plt.show()
