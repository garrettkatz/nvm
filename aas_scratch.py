import numpy as np
import matplotlib.pyplot as plt
from tokens import N_LAYER, LAYERS, DEVICES, TOKENS, PATTERNS, get_token
from gates import default_gates, N_GH, PAD
from flash_rom import cpu_state, V_START, V_READY
from aas_nvm import WEIGHTS, tick, print_state, state_string, weight_update

np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

REG_INIT = {}
REG_INIT = {"TC": "NULL"}
program = [ # label opc op1 op2 op3
    "NULL","SET","REG2","RIGHT","NULL", # Store right gaze for comparison with TC
    "NULL","SET","REG3","LEFT","NULL", # Store left gaze for comparison with TC
    "LOOP","SET","FEF","CENTER","NULL", # Fixate on center
    "GAZE","CMP","REG1","TC","REG2", # Check if TC detects rightward gaze
    "NULL","JMP","REG1","LOOK","NULL", # If so, skip to saccade step
    "NULL","CMP","REG1","TC","REG3", # Check if TC detects leftward gaze
    "NULL","JMP","REG1","LOOK","NULL", # If so, skip to saccade step
    "NULL","SET","REG1","TRUE","NULL", # If here, gaze not known yet, prepare unconditional jump
    "NULL","JMP","REG1","GAZE","NULL", # Check for gaze again
    "LOOK","MOV","FEF","TC","NULL", # TC detected gaze, overwrite FEF with gaze direction
    "NULL","RET","NULL","NULL","NULL", # Successful saccade, terminate program
]

# encode program transits
V_PROG = PAD*np.sign(np.concatenate(tuple(TOKENS[t] for t in program), axis=1)) # program
V_PROG = np.concatenate((V_PROG, PAD*np.sign(np.random.randn(*V_PROG.shape))),axis=0) # add hidden

# link labels
labels = {program[p]:p for p in range(0,len(program),5) if program[p] != "NULL"}
for p in range(3,len(program),5):
    if program[p] in labels:
        V_PROG[:N_LAYER,p+1] = V_PROG[N_LAYER:, labels[program[p]]]

# flash ram with program memory
X, Y = V_PROG[N_LAYER:,:-1], V_PROG[:,1:]

do_global = False
if do_global:
    # global
    W_RAM = np.linalg.lstsq(X.T, np.arctanh(Y).T, rcond=None)[0].T
else:
    # local
    # W_RAM = np.arctanh(Y).dot(X.T) / N_LAYER #/ (N_LAYER*PAD**2)
    W_RAM = np.zeros((2*N_LAYER, N_LAYER))
    for p in range(V_PROG.shape[1]-1):
        W_RAM = weight_update(W_RAM, V_PROG[N_LAYER:,[p]], V_PROG[:,[p+1]])

print("Flash ram residual max: %f"%np.fabs(Y - np.tanh(W_RAM.dot(X))).max())
print("Flash ram residual mad: %f"%np.fabs(Y - np.tanh(W_RAM.dot(X))).mean())
print("Flash ram sign diffs: %d"%(np.sign(Y) != np.sign(np.tanh(W_RAM.dot(X)))).sum())
raw_input('continue?')

# ram
WEIGHTS[("MEM","MEMH")] = W_RAM[:N_LAYER,:]
WEIGHTS[("MEMH","MEMH")] = W_RAM[N_LAYER:,:]

# initialize cpu activity
ACTIVITY = {k: -PAD*np.ones((N_LAYER,1)) for k in LAYERS+DEVICES}
ACTIVITY["GATES"] = V_START[:N_GH,:] 
ACTIVITY["MEM"] = V_PROG[:N_LAYER,[0]]
ACTIVITY["MEMH"] = V_PROG[N_LAYER:,[0]]
ACTIVITY["CMPA"] = PAD*np.sign(np.random.randn(N_LAYER,1))
ACTIVITY["CMPB"] = -ACTIVITY["CMPA"]
for k,v in REG_INIT.items(): ACTIVITY[k] = TOKENS[v] 

# run nvm loop
HISTORY = [ACTIVITY]
show_each = False
for t in range(1000):
    if show_each:
        if t % 2 == 0:
            print("tick %d:"%t)
            if (ACTIVITY["GATES"] * V_READY[:N_GH,:] >= 0).all():
                print("Ready to execute instruction")
            print_state(ACTIVITY)
            if get_token(ACTIVITY["OPC"]) == "RET": break
    else:
        if (ACTIVITY["GATES"] * V_READY[:N_GH,:] >= 0).all():
            print("tick %3d: %s"%(t,state_string(ACTIVITY)))
        if t % 2 == 0 and get_token(ACTIVITY["OPC"]) == "RET":
            print("tick %3d: %s"%(t,state_string(ACTIVITY)))
            break
    
    ACTIVITY = tick(ACTIVITY, WEIGHTS)
    HISTORY.append(ACTIVITY)


A_LAYERS = ["MEM","MEMH","GATES","OPC","OP1","OP2","OP3","CMPA","CMPB","CMPH","CMPO","REG1","REG2","REG3","FEF","TC"]

A = np.zeros((N_GH + (len(A_LAYERS)-1)*N_LAYER,len(HISTORY)))
mx = np.zeros(len(HISTORY))
for h in range(len(HISTORY)):
    A[:,[h]] = np.concatenate([HISTORY[h][k] for k in A_LAYERS],axis=0)
    mx[h] = np.concatenate([HISTORY[h][k] for k in A_LAYERS if k[:3] != "CMP"],axis=0).max()
# mx = np.fabs(A).max(axis=0)
# print(mx)
print((mx.min(), mx.mean(), mx.max()))






















xt = (np.arange(A.shape[1]) + .5)[::int(A.shape[1]/10)]
xl = np.array(["%d"%t for t in range(A.shape[1])])[::int(A.shape[1]/10)]
yt = np.array([HISTORY[0][k].shape[0] for k in A_LAYERS])
yt = yt.cumsum() - yt/2

plt.figure()
plt.imshow(A, cmap='gray', vmin=-1, vmax=1, aspect='auto')
plt.xticks(xt, xl)
plt.yticks(yt, A_LAYERS)
plt.show()
