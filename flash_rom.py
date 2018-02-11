import numpy as np
import matplotlib.pyplot as plt
from tokens import LAYERS, CPU_LAYERS, USER_LAYERS, DEVICES, N_LAYER, TOKENS, get_token
from gates import N_GATES, N_HGATES, N_GH, PAD, GATE_INDEX, default_gates, get_open_gates

np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .0f'%x})

def cpu_state(ungate=[], hidden=None, opc=None, op1=None, op2=None, op3=None):
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
    gates[N_GATES:,:] = hidden
    # op
    opc = TOKENS[opc] if opc is not None else np.zeros((N_LAYER,1))
    op1 = TOKENS[op1] if op1 is not None else np.zeros((N_LAYER,1))
    op2 = TOKENS[op2] if op2 is not None else np.zeros((N_LAYER,1))
    op3 = TOKENS[op3] if op3 is not None else np.zeros((N_LAYER,1))
    # pad result in hypercube
    return PAD*np.sign(np.concatenate((gates, opc, op1, op2, op3), axis=0))

def with_ops(v, opc=None, op1=None, op2=None, op3=None):
    """Copy of cpu state v except with potentially different instruction"""
    gates = v[:N_GH,[0]]
    opc = TOKENS[opc] if opc is not None else np.zeros((N_LAYER,1))
    op1 = TOKENS[op1] if op1 is not None else np.zeros((N_LAYER,1))
    op2 = TOKENS[op2] if op2 is not None else np.zeros((N_LAYER,1))
    op3 = TOKENS[op3] if op3 is not None else np.zeros((N_LAYER,1))
    # pad result in hypercube
    return PAD*np.sign(np.concatenate((gates, opc, op1, op2, op3), axis=0))

def cpu_decode(v):
    """Decode cpu state vector into human-readable dict"""
    return {
        "GATES": get_open_gates(v[:N_GATES,[0]]),
        "OPC": get_token(v[N_GH+0*N_LAYER:N_GH+1*N_LAYER,[0]]),
        "OP1": get_token(v[N_GH+1*N_LAYER:N_GH+2*N_LAYER,[0]]),
        "OP2": get_token(v[N_GH+2*N_LAYER:N_GH+3*N_LAYER,[0]]),
        "OP3": get_token(v[N_GH+3*N_LAYER:N_GH+4*N_LAYER,[0]]),
    }

def add_transit(X, Y, cpu_old, cpu_new):
    """
    Add transition cpu_old, cpu_new to lists X, Y respectively
    Return cpu_new for "chaining"
    cpu_old should have open (g,op,u) if and only if op is nonzero
    """
    opregs = ["OPC","OP1","OP2","OP3"]
    open_gates = get_open_gates(cpu_old)
    for o in range(len(opregs)):
        if (("GATES",opregs[o],"U") in open_gates) == (cpu_old[N_GH+o*N_LAYER:N_GH+(o+1)*N_LAYER,:]==0).all():
            print("corrupted transition! old ~~> new:")
            print(cpu_decode(cpu_old))
            print(cpu_decode(cpu_new))
            raise Exception("Corrupted transition!")
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
                print("Non-determinism at X[:,%d|%d]!"%(j,k))
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
        np.zeros((4*N_LAYER,A.shape[1])) # ops unavailable when gates are closed again
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
    XZ = np.concatenate((X, Z), axis=1)
    ZY = np.concatenate((Z[:N_GH,:], Y[:N_GH,:]), axis=1)
    W = np.linalg.lstsq(XZ.T, np.arctanh(ZY).T, rcond=None)[0].T    
    if verbose:
        print("W shape:")
        print(W.shape)
        print("Flash residual = %f"%np.fabs(ZY - np.tanh(W.dot(XZ))).max())

    return W, Z

# gates for inter-layer copy from f to t
cop = lambda t, f: [(t,f,"U"),(t,t,"C")]

# gates for full memory (value and hidden) layer updates
memu = [("MEM"+a,"MEM"+b,"U") for a in ["","H"] for b in ["","H"]]

######## FLASH ROM #########

X, Y = [], [] # growing lists of transitions

###### Load instruction from mem into cpu, one operand at a time

V_START = cpu_state(hidden = -PAD*np.sign(np.random.randn(N_HGATES,1)),ungate = memu)
v = V_START
for reg in ["OPC","OP1","OP2","OP3"]:
    v = add_transit(X, Y, v, cpu_state(ungate = memu + cop(reg,"MEM")))

# Let opcode bias the gate layer
v = add_transit(X, Y, v, cpu_state(ungate = [("GATES","OPC","U")]))

# Ready to execute instruction
v_ready = v.copy()

###### NOP instruction

add_transit(X, Y, with_ops(v_ready, opc="NOP"), V_START) # begin next clock cycle

###### SET instruction

# Let op1 bias the gate layer
v_inst = add_transit(X, Y, with_ops(v_ready,opc="SET"), cpu_state(ungate = [("GATES","OP1","U")]))

# Add transits for each op1 possibility
for to_layer in USER_LAYERS + DEVICES:
    # copy value in op2 to layer in op1
    v = add_transit(X, Y, with_ops(v_inst, op1=to_layer), cpu_state(ungate=cop(to_layer,"OP2")))
    # begin next clock cycle
    add_transit(X, Y, v, V_START)

###### MOV instruction

# Let op1,op2 bias the gate layer
v_inst = add_transit(X, Y, with_ops(v_ready,opc="MOV"),
    cpu_state(ungate = [("GATES","OP1","U"),("GATES","OP2","U")]))

# Add transits for each op1,op2 possibility
for to_layer in USER_LAYERS + DEVICES:
    for from_layer in USER_LAYERS + DEVICES:
        # copy from layer in op2 to layer in op1
        v = add_transit(X, Y, with_ops(v_inst, op1=to_layer, op2=from_layer),
            cpu_state(ungate=cop(to_layer, from_layer)))
        # begin next clock cycle
        add_transit(X, Y, v, V_START)

###### CMP instruction

# Let compare ops bias the gate layer
v_inst = add_transit(X, Y, with_ops(v_ready,opc="CMP"),
    cpu_state(ungate = [("GATES","OP2","U"),("GATES","OP3","U")]))

# # Prepare for result op to bias the gate layer after compare ops copied
v_comp = cpu_state(ungate = [("GATES","OP1","U")])

# Add transits for compare ops possibility
for a_layer in USER_LAYERS + DEVICES:
    for b_layer in USER_LAYERS + DEVICES:
        # copy a,b layers into compare circuit
        v = add_transit(X, Y, with_ops(v_inst, op2=a_layer, op3=b_layer),
            cpu_state(ungate= cop("CMPA",a_layer) + cop("CMPB",b_layer)))
        # prepare to copy result
        add_transit(X, Y, v, v_comp)
        # when in state v_comp, cmph is up to date

# Add transits for result op possibilities
for result_layer in USER_LAYERS + DEVICES:
    v = add_transit(X, Y, with_ops(v_comp, op1=result_layer),
        cpu_state(ungate= cop(result_layer,"CMPO")))
    # when in state v, cmpo is up to date
    # begin next clock cycle
    add_transit(X, Y, v, V_START)

###### JMP instruction

# Let op1 bias the gate layer
v_inst = add_transit(X, Y, with_ops(v_ready,opc="JMP"), cpu_state(ungate = [("GATES","OP1","U")]))

for jmp_layer in USER_LAYERS+DEVICES:
    # Overwrite op1 with the layer it names
    v = add_transit(X, Y, with_ops(v_inst, op1=jmp_layer),
        cpu_state(ungate=cop("OP1",jmp_layer)))
    # Let copied value bias gate again
    v_jmp = add_transit(X, Y, v, cpu_state(ungate = [("GATES","OP1","U")]))
    # If false, don't jump
    add_transit(X, Y, with_ops(v_jmp, op1="FALSE"), V_START)
    # If true, do jump
    v = add_transit(X, Y, with_ops(v_jmp, op1="TRUE"), 
        cpu_state(ungate=cop("MEM","OP2")+cop("MEMH","OP3")))
    add_transit(X, Y, v, V_START)

###### Flash to ROM
X = np.concatenate(X,axis=1)
Y = np.concatenate(Y,axis=1)
W_ROM, Z = flash_rom(X, Y, verbose=True)

# do_pause = True

# if not do_pause: plt.ion()

# kr = 1
# plt.subplot(1,4,1)
# plt.imshow(np.kron((X-X.min())/(X.max()-X.min()),np.ones((1,kr))), cmap='gray')
# plt.subplot(1,4,2)
# plt.imshow(np.kron((Z-Z.min())/(Z.max()-Z.min()),np.ones((1,kr))), cmap='gray')
# plt.subplot(1,4,3)
# plt.imshow(np.kron((Y-Y.min())/(Y.max()-Y.min()),np.ones((1,kr))), cmap='gray')
# plt.subplot(1,4,4)
# plt.imshow((W_ROM-W_ROM.min())/(W_ROM.max()-W_ROM.min()), cmap='gray')
# plt.show()

# if not do_pause: plt.ioff()
