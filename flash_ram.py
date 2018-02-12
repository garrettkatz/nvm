import numpy as np
import matplotlib.pyplot as plt
from tokens import N_LAYER, LAYERS, DEVICES, TOKENS, PATTERNS, add_token, get_token
from gates import PAD

np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

program = [ # label opc op1 op2 op3
    "NULL","SET","REG2","RIGHT","NULL", # Store right gaze for comparison with TC
    "NULL","SET","REG3","LEFT","NULL", # Store left gaze for comparison with TC
    "LOOP","SET","FEF","CENTER","NULL", # Fixate on center
    "GAZE","CMP","REG1","TC","REG2", # Check if TC detects rightward gaze
    "NULL","JMP","REG1","LOOP","NULL", #!!!!!!!!!!CHANGES If so, skip to saccade step
    "NULL","CMP","REG1","TC","REG3", # Check if TC detects leftward gaze
    "NULL","JMP","REG1","LOOK","NULL", # If so, skip to saccade step
    "NULL","SET","REG1","TRUE","NULL", # If here, gaze not known yet, prepare unconditional jump
    "NULL","JMP","REG1","GAZE","NULL", # Check for gaze again
    "LOOK","MOV","FEF","TC","NULL", # TC detected gaze, overwrite FEF with gaze direction
    "NULL","RET","NULL","NULL","NULL", # Successful saccade, terminate program
]
program = program[:25]
P = len(program)

# encode program transits
V_PROG = PAD*np.sign(np.concatenate(tuple(TOKENS[t] for t in program), axis=1)) # program
V_PROG = np.concatenate((V_PROG, PAD*np.sign(np.random.randn(*V_PROG.shape))),axis=0) # add hidden

# 1-many token transits
token_transits = {}
for p in range(P-1):
    if program[p] not in token_transits:
        token_transits[program[p]] = set()
    token_transits[program[p]].add(program[p+1])

# link labels
labels = {program[p]:p for p in range(0,P,5) if program[p] != "NULL"}
for p in range(3,P,5):
    if program[p] in labels:
        hidden = V_PROG[N_LAYER:, labels[program[p]]]
        V_PROG[:N_LAYER,p+1] = hidden
        add_token(program[p]+"H", hidden)

def perf(W, X, Y):
    fX = np.tanh(W.dot(X))
    resid = np.sqrt(np.mean((Y - fX)**2, axis=0))
    vdiff = ((Y[:N_LAYER,:] * fX[:N_LAYER,:]) < 0).sum(axis=0)
    hdiff = ((Y[N_LAYER:,:] * fX[N_LAYER:,:]) < 0).sum(axis=0)
    diff = vdiff + hdiff
    print("residuals:")
    print(resid)
    print("sign diffs (out of %d): v,h,both"%X.shape[0])
    print(vdiff)
    print(hdiff)
    print(diff)
    return resid, vdiff, hdiff, diff

def learn_global(X, Y):
    W = np.linalg.lstsq(X.T, np.arctanh(Y).T, rcond=None)[0].T
    return W

def learn_ahebb(X, Y):
    N = X.shape[0]
    W = np.zeros((N,N))
    for t in range(X.shape[1]):
        W += np.arctanh(Y[:,[t]]) * X[:,[t]].T/(N*PAD**2)
    return W

def learn_ahebb_h(X, Y):
    """no v->_ connections, only h->_"""
    W = np.zeros((2*N_LAYER,2*N_LAYER))
    W[:N_LAYER,N_LAYER:] = learn_ahebb(X[N_LAYER:,:], Y[:N_LAYER,:])
    W[N_LAYER:,N_LAYER:] = learn_ahebb(X[N_LAYER:,:], Y[N_LAYER:,:])
    return W

# learn = learn_global
# learn = learn_ahebb
learn = learn_ahebb_h

# flash ram with program memory
X, Y = V_PROG[:,:-1], V_PROG[:,1:]
W_RAM = learn(X, Y)
resid, vdiff, hdiff, diff = perf(W_RAM, X, Y)
print("token transit counts:")
ttc = np.array([len(token_transits[p]) for p in program[:-1]])
print(ttc)
# print('cc=%f'%np.corrcoef(ttc, vdiff))
print(np.corrcoef(ttc, vdiff))
# print("PROG X shape:")
# print(X.shape)
# W_RAM = np.arctanh(Y).dot(X.T) # local

V = np.zeros(V_PROG.shape)
v = V_PROG[:,[0]]
V[:,[0]] = v
for t in range(1,P):
    v = np.tanh(W_RAM.dot(v))
    V[:,[t]] = v

kr = 3*N_LAYER/P
xt = np.arange(P)*kr + kr/2
xl1 = np.array(["%d\n%s"%(p,get_token(V_PROG[:N_LAYER,[p]])) for p in range(P)])
xl2 = np.array(["%d\n%s"%(p,get_token(V[:N_LAYER,[p]])) for p in range(P)])

print(" ".join(["%s"%(get_token(V_PROG[:N_LAYER,[p]])) for p in range(P)]))
print(" ".join(["%s"%(get_token(V[:N_LAYER,[p]])) for p in range(P)]))

# plt.figure()
# plt.subplot(2,1,1)
# plt.imshow(np.kron(V_PROG,np.ones((1,kr))), cmap='gray', vmin=-1, vmax=1)
# plt.xticks(xt, xl1)
# plt.subplot(2,1,2)
# plt.imshow(np.kron(V,np.ones((1,kr))), cmap='gray', vmin=-1, vmax=1)
# plt.xticks(xt, xl2)
# plt.tight_layout()
# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
# plt.show()
