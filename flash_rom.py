import numpy as np
import matplotlib.pyplot as plt
from tokens import LAYERS, DEVICES, N_LAYER, TOKENS, get_token
from gates import N_GATES, N_HGATES, N_GH, PAD, GATE_INDEX, default_gates

np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .0f'%x})

def cpu_state(ungate=[], hidden=None, opc=None, op1=None, op2=None):
    """
    Convert human-readable cpu state to activity vector.
    ungate is a list of (to,from,mode) gate keys
    hidden is a hidden activity vector
    opc, op1, op2 are the opcode and operands
    """
    # gates
    gates = default_gates()
    for g in ungate: gates[GATE_INDEX[g],0] = 1
    # hidden
    if hidden is None: hidden = np.random.randn(N_HGATES,1)
    # op
    opc = TOKENS[opc] if opc is not None else np.zeros((N_LAYER,1))
    op1 = TOKENS[op1] if op1 is not None else np.zeros((N_LAYER,1))
    op2 = TOKENS[op2] if op2 is not None else np.zeros((N_LAYER,1))
    # pad result in hypercube
    return PAD*np.sign(np.concatenate((gates, hidden, opc, op1, op2), axis=0))

def add_transit(X, Y, cpu_old, cpu_new):
    """
    Add transition cpu_old, cpu_new to lists X, Y respectively
    Return cpu_new for "chaining"
    """
    X.append(cpu_old)
    Y.append(cpu_new)
    return cpu_new.copy()
    
def check_deterministic(X, Y):
    """
    Check for deterministic X ~~> Y transitions (identical X need identical Y)
    """
    for j in range(X.shape[1]):
        for k in range(X.shape[1]):
            if (X[:,j]==X[:,k]).all() and (Y[:,j]!=Y[:,k]).any():
                return False
    return True
    
def flash_rom(X, Y, verbose=True):
    """
    Construct W that transitions states in X to corresponding states in Y
    X, Y are arrays, with cpu activity states as columns
    To deal with low-rank X, each transition uses an intermediate hidden step
    """

    # for low-rank X, get coefficients A of X's column space
    _, sv, A = np.linalg.svd(X, full_matrices=False)
    rank_tol = sv.max() * max(X.shape) * np.finfo(sv.dtype).eps # from numpy
    A = A[sv > rank_tol, :]
    
    # use A to set intermediate Z that is low-rank pre non-linearity
    Z = np.concatenate((
        default_gates()[:N_GATES]*np.ones((N_GATES, A.shape[1])),
        np.tanh(np.random.randn(N_HGATES, A.shape[0]).dot(A)),
        X[N_GH:,:]
    ), axis=0)

    if verbose:
        print("!! Flashing !!")
        print("Deterministic?")
        print(check_deterministic(X,Y))
        print("X,Z,Y shapes and ranks:")
        for M in [X, Z, Y]:
            print(M.shape)
            print(np.linalg.matrix_rank(M))

    # solve linear equations
    X = np.concatenate((X, Z), axis=1)
    Y = np.concatenate((Z[:N_GH,:], Y[:N_GH,:]), axis=1)
    W = np.linalg.lstsq(X.T, np.arctanh(Y).T, rcond=None)[0].T    
    if verbose:
        print("Flash residual = %f"%np.fabs(Y - np.tanh(W.dot(X))).max())

    return W

# gates for inter-layer copy from f to t
cop = lambda t, f: [(t,f,"U"),(t,f,"C")]

# gates for full memory (value and hidden) layer updates
memu = [("MEM"+a,"MEM"+b,"U") for a in ["1","2"] for b in ["1","2"]]

# operation registers
opregs = ["OPCODE","OPERAND1","OPERAND2"]

######## FLASH ROM #########

X, Y = [], [] # growing lists of transitions
v0 = cpu_state() # beginning of clock cycle

### Load instructions from mem into cpu, one operand at a time
v = v0
for reg in opregs:
    v = add_transit(X, Y, v, cpu_state(ungate = memu + cop(reg,"MEM1")))
v = add_transit(X, Y, v, cpu_state(ungate = [("GATES",reg,"U") for reg in opregs]))

### ready to execute instruction
v_ready = v 

# ### SET instruction
# v1_set = 
# add_transit(X, Y, v0, cpu_state(
#     gates

### Return to beginning state for next clock cycle
add_transit(X, Y, v, v0)

### Flash to ROM
X = np.concatenate(X,axis=1)
Y = np.concatenate(Y,axis=1)

W = flash_rom(X, Y, verbose=True)

plt.subplot(1,3,1)
plt.imshow(np.kron((X-X.min())/(X.max()-X.min()),np.ones((1,3))), cmap='gray')
plt.subplot(1,3,2)
plt.imshow(np.kron((Y-Y.min())/(Y.max()-Y.min()),np.ones((1,3))), cmap='gray')
plt.subplot(1,3,3)
plt.imshow((W-W.min())/(W.max()-W.min()), cmap='gray')
plt.show()
