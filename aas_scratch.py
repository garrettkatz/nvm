import sys
import numpy as np
import matplotlib.pyplot as plt
from tokens import N_LAYER, LAYERS, DEVICES, TOKENS, PATTERNS, get_token
from gates import default_gates, N_GH, PAD
from flash_rom import cpu_state, V_START, V_READY
from aas_nvm import make_weights, store_program, tick, print_state, state_string, weight_update

np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

WEIGHTS = make_weights()

REG_INIT = {}
REG_INIT = {"TC": "NULL"}
program = [ # label opc op1 op2 op3
    "NULL","SET","REG2","RIGHT","NULL", # Store right gaze for comparison with TC
    "NULL","SET","REG3","LEFT","NULL", # Store left gaze for comparison with TC
    "NULL","SET","FEF","CENTER","NULL", # Fixate on center
    "LOOP","CMP","REG1","TC","REG2", # Check if TC detects rightward gaze
    "NULL","JMP","REG1","LOOK","NULL", # If so, skip to saccade step
    "NULL","CMP","REG1","TC","REG3", # Check if TC detects leftward gaze
    "NULL","JMP","REG1","LOOK","NULL", # If so, skip to saccade step
    "NULL","SET","REG1","TRUE","NULL", # If here, gaze not known yet, prepare unconditional jump
    "NULL","JMP","REG1","LOOP","NULL", # Check for gaze again
    "LOOK","MOV","FEF","TC","NULL", # TC detected gaze, overwrite FEF with gaze direction
    "NULL","RET","NULL","NULL","NULL", # Successful saccade, terminate program
]

program = [ # label opc op1 op2 op3
    "NULL","SET","REG2","FACE","NULL", # Store "face" flag for comparison with TC
    "NULL","SET","REG3","TRUE","NULL", # Store "true" for unconditional jumps
    # plan center saccade
    "REPEAT","SET","FEF","CENTER","NULL",
    # initiate saccade
    "NULL","SET","SC","SACCADE","NULL",
    "NULL","SET","SC","OFF","NULL",
    # wait face
    "LOOP","CMP","REG1","TC","REG2", # Check if TC detects face
    "NULL","JMP","REG1","LOOK","NULL", # If so, skip to saccade step
    "NULL","JMP","REG3","LOOP","NULL", # Check for face again
    # initiate saccade
    "LOOK","SET","SC","SACCADE","NULL", # TC detected gaze, allow saccade
    "NULL","SET","SC","OFF","NULL", # TC detected gaze, allow saccade
    # repeat
    "NULL","JMP","REG3","REPEAT","NULL", # Successful saccade, repeat
]

WEIGHTS, v_prog = store_program(WEIGHTS, program, do_global=True)

# initialize cpu activity
ACTIVITY = {k: -PAD*np.ones((N_LAYER,1)) for k in LAYERS+DEVICES}
ACTIVITY["GATES"] = V_START[:N_GH,:] 
ACTIVITY["MEM"] = v_prog[:N_LAYER,[0]]
ACTIVITY["MEMH"] = v_prog[N_LAYER:,[0]]
ACTIVITY["CMPA"] = PAD*np.sign(np.random.randn(N_LAYER,1))
ACTIVITY["CMPB"] = -ACTIVITY["CMPA"]
for k,v in REG_INIT.items(): ACTIVITY[k] = TOKENS[v] 

# run nvm loop
HISTORY = [ACTIVITY]
ready_t = []
show_each = False
for t in range(750):
    if show_each:
        if t % 2 == 0:
            print("tick %d:"%t)
            if (ACTIVITY["GATES"] * V_READY[:N_GH,:] >= 0).all():
                print("Ready to execute instruction")
                ready_t.append(t)
            print_state(ACTIVITY)
            if get_token(ACTIVITY["OPC"]) == "RET": break
    else:
        if (ACTIVITY["GATES"] * V_READY[:N_GH,:] >= 0).all():
            print("tick %3d: %s"%(t,state_string(ACTIVITY)))
            ready_t.append(t)
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

# xt = (np.arange(A.shape[1]) + .5)[::int(A.shape[1]/10)]
# xl = np.array(["%d"%t for t in range(A.shape[1])])[::int(A.shape[1]/10)]
xt = ready_t
xl = []
for t in ready_t:
    ops = []
    # for op in ["MEM","OPC","OP1","OP2","OP3"]:
    for op in ["OPC","OP1","OP2","OP3"]:
        tok = get_token(HISTORY[t][op])
        ops.append("" if tok in ["NULL","?"] else tok)
    xl.append("\n".join([str(t)]+ops))
yt = np.array([HISTORY[0][k].shape[0] for k in A_LAYERS])
yt = yt.cumsum() - yt/2

plt.figure()
plt.imshow(A, cmap='gray', vmin=-1, vmax=1, aspect='auto')
plt.xticks(xt, xl)
plt.yticks(yt, A_LAYERS)
plt.show()
