import numpy as np
import matplotlib.pyplot as plt
from tokens import N_LAYER, LAYERS, DEVICES, TOKENS, PATTERNS, get_token
from gates import default_gates, N_GH, PAD
from flash_rom import W_ROM, cpu_state, V_START
from aas_nvm import tick, print_state

np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

program = [
    "NOP", "NULL", "NULL",
    "SET", "FEF", "CENTER",
    "SET", "COMPARE1", "LEFT",
    "LOAD", "COMPARE2", "TC",
    # "LOAD", "REGISTER1", "COMPARE",
    # "IF", "REGISTER1", "LLEFT",
    # "LOAD", "COMPARE2", "RIGHT",
    # "LOAD", "REGISTER1", "COMPARE",
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

V_PROG = PAD*np.sign(np.concatenate(tuple(TOKENS[t] for t in program), axis=1)) # program
V_PROG = np.concatenate((V_PROG, PAD*np.sign(np.random.randn(*V_PROG.shape))),axis=0) # add hidden

# flash ram
X, Y = V_PROG[:,:-1], V_PROG[:,1:]
W_RAM = np.linalg.lstsq(X.T, np.arctanh(Y).T, rcond=None)[0].T
print("Flash ram residual: %f"%np.fabs(Y - np.tanh(W_RAM.dot(X))).max())

# dict of inter/intra layer weight matrices
WEIGHTS = {}
# relays
for to_layer in LAYERS + DEVICES:
    for from_layer in LAYERS + DEVICES:
        WEIGHTS[(to_layer,from_layer)] = np.eye(N_LAYER,N_LAYER) * np.arctanh(PAD)/PAD
# ram
WEIGHTS[("MEM1","MEM1")] = W_RAM[:N_LAYER,:N_LAYER]
WEIGHTS[("MEM1","MEM2")] = W_RAM[:N_LAYER,N_LAYER:]
WEIGHTS[("MEM2","MEM1")] = W_RAM[N_LAYER:,:N_LAYER]
WEIGHTS[("MEM2","MEM2")] = W_RAM[N_LAYER:,N_LAYER:]
# rom
WEIGHTS[("GATES","GATES")] = W_ROM[:,:N_GH]
WEIGHTS[("GATES","OPCODE")] = W_ROM[:,N_GH+0*N_LAYER:N_GH+1*N_LAYER]
WEIGHTS[("GATES","OPERAND1")] = W_ROM[:,N_GH+1*N_LAYER:N_GH+2*N_LAYER]
WEIGHTS[("GATES","OPERAND2")] = W_ROM[:,N_GH+2*N_LAYER:N_GH+3*N_LAYER]


# initialize activity
ACTIVITY = {k: -PAD*np.ones((N_LAYER,1)) for k in LAYERS+DEVICES}
ACTIVITY["GATES"] = V_START[:N_GH,:] 
ACTIVITY["MEM1"] = V_PROG[:N_LAYER,[0]]
ACTIVITY["MEM2"] = V_PROG[N_LAYER:,[0]]
ACTIVITY["TC"] = TOKENS["RIGHT"]

# run nvm
HISTORY = [ACTIVITY]
for t in range(50):
    # if t % 2 == 0:
    #     print("tick %d:"%t)
    #     print_state(ACTIVITY)
    #     if get_token(ACTIVITY["OPCODE"]) == "RET": break
    if t % 2 == 0 and get_token(ACTIVITY["OPCODE"]) == "RET":
        print("tick %d:"%t)
        print_state(ACTIVITY)
        break
    
    ACTIVITY = tick(ACTIVITY, WEIGHTS)
    HISTORY.append(ACTIVITY)

if not get_token(ACTIVITY["OPCODE"]) == "RET":
    print("tick %d:"%t)
    print_state(ACTIVITY)

A = np.zeros((N_GH + 5*N_LAYER,len(HISTORY)))
for h in range(len(HISTORY)):
    A[:,[h]] = np.concatenate((
        HISTORY[h]["GATES"],
        HISTORY[h]["OPCODE"],
        HISTORY[h]["OPERAND1"],
        HISTORY[h]["OPERAND2"],
        HISTORY[h]["FEF"],
        HISTORY[h]["TC"],
    ),axis=0)

mx = np.fabs(A).max(axis=0)
print(mx)
print((mx.min(), mx.mean(), mx.max()))

plt.figure()
kr = 3
plt.imshow(np.kron((A-A.min())/(A.max()-A.min()),np.ones((1,kr))), cmap='gray')
plt.show()
