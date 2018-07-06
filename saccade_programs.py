import numpy as np
import matplotlib.pyplot as plt
from layer import Layer
from coder import Coder
from gate_map import make_nvm_gate_map
from activator import *
from learning_rules import *
from nvm_assembler import assemble
from nvm_linker import link
from nvm_net import NVMNet

# New version using pef
aas_program = {"aas":"""

loop:   mov fef center
        mov sc on
        mov sc off
cross:  mov pef tc
        jmp pef
left:   mov fef right
        jmp look
right:  mov fef left
        jmp look
look:   mov sc on
        mov sc off
        jmp loop
        exit

"""}

# # New version that uses comparison
# aas_program = {"aas":"""

# loop:   mov fef center
#         mov sc on
#         mov sc off
# wait:   cmp tc cross
#         jie cross
#         jmp wait
# cross:  jmp tc
# left:   mov fef right
#         jmp look
# right:  mov fef left
#         jmp look
# look:   mov sc on
#         mov sc off
#         jmp loop
#         exit

# """}

# Old version that doesn't depend on comparison
'''
aas_program = {"aas":"""

loop:   mov fef center
        mov sc on
        mov sc off
wait:   jmp tc
cross:  jmp tc
# cross:  jmp tc
left:   mov fef right
        jmp look
right:  mov fef left
        jmp look
look:   mov sc on
        mov sc off
        jmp loop
        exit

"""}
'''

def make_ef(name, pad, activator, rows, columns):
    dim = min(rows, columns)

    act = activator(pad, rows*columns)
    ef_coder = Coder(act)

    Y, X = np.mgrid[:rows,:columns] # transpose for bitmap
    R = .1
    # # Cts
    # center = act.off + (act.on-act.off)*np.exp(-((X-.5*columns)**2 + (Y-.5*rows)**2)/(R*dim)**2)
    # left = act.off + (act.on-act.off)*np.exp(-((X-.0*columns)**2 + (Y-.5*rows)**2)/(R*dim)**2)
    # right = act.off + (act.on-act.off)*np.exp(-((X-1.*columns)**2 + (Y-.5*rows)**2)/(R*dim)**2)
    # Binary
    center = act.off + (act.on-act.off)*((X-.5*columns)**2 + (Y-.5*rows)**2 < (R*dim)**2)
    left = act.off + (act.on-act.off)*((X-.125*columns)**2 + (Y-.5*rows)**2 < (R*dim)**2)
    right = act.off + (act.on-act.off)*((X-.875*columns)**2 + (Y-.5*rows)**2 < (R*dim)**2)

    ef_coder.encode("cross", center.reshape((rows*columns,1)))
    ef_coder.encode("center", center.reshape((rows*columns,1)))
    ef_coder.encode("left", left.reshape((rows*columns,1)))
    ef_coder.encode("right", right.reshape((rows*columns,1)))

    return Layer(name, (rows,columns), act, ef_coder)

def make_sc(pad, activator, rows, columns):

    act = activator(pad, rows*columns)
    sc_coder = Coder(act)
    sc_coder.encode("on", act.on*np.ones((rows*columns,1)))
    sc_coder.encode("off", act.off*np.ones((rows*columns,1)))
    return Layer("sc", (rows,columns), act, sc_coder)

def make_saccade_nvm(activator_label):

    # set up activator
    if activator_label == "logistic":
        activator = logistic_activator
    if activator_label == "tanh":
        activator = tanh_activator
    learning_rule = hebbian

    # make network
    layer_shape = (32,32)
    layer_size = layer_shape[0]*layer_shape[1]
    pad = 0.0001
    act = activator(pad, layer_size)
    
    devices = {
        "tc": Layer("tc", layer_shape, act, Coder(act)),
        "fef": make_ef("fef", pad, activator, 68, 96),
        "pef": make_ef("pef", pad, activator, 68, 96),
        "sc": make_sc(pad, activator, 5, 5)}

    shapes = {"gh": (32,16)}
    nvmnet = NVMNet(layer_shape, pad, activator, learning_rule, devices, shapes=shapes)

    # assemble and link programs
    for name, program in aas_program.items():
        nvmnet.assemble(program, name, verbose=1)
    nvmnet.link(verbose=2)

    # redo pef/fef linkages with special learning rule
    # pef -> ip
    X = np.concatenate([nvmnet.layers["pef"].coder.encode(tok)
        for tok in ["left","right","cross"]], axis=1)
    Y = np.concatenate([nvmnet.layers["ip"].coder.encode(tok)
        for tok in ["left","right","cross"]], axis=1)
    nvmnet.weights[("ip","pef")] = nvmnet.layers["ip"].activator.g(Y).dot(
        X.T / (X**2).sum(axis=0)[:, np.newaxis])
    nvmnet.biases[("ip","pef")][:] = 0

    # initialize layers
    nvmnet.activity["ip"] = nvmnet.layers["ip"].coder.encode(name) # program pointer
    nvmnet.activity["tc"] = nvmnet.layers["tc"].coder.encode("cross") # waiting for face

    return nvmnet

if __name__ == "__main__":
    
    nvmnet = make_saccade_nvm("logistic")
    raw_input("continue?")
    
    show_layers = [
        ["go", "gh", "gi", "ip"] + ["op"+x for x in "c12"] + nvmnet.devices.keys(),
    ]
    show_tokens = True
    show_corrosion = True
    show_gates = False

    history = []
    start_t = []
    int_sched = [20, 20]
    tc_sched = [60, 110, 200]
    for t in range(tc_sched[2]):

        # brief interrupt
        if t >= int_sched[0]:
            nvmnet.activity["gi"] = nvmnet.layers["gi"].coder.encode("pause")
        if t >= int_sched[1]:
            nvmnet.activity["gi"] = nvmnet.layers["gi"].coder.encode("quiet")

        ### occassionally change tc
        if t > 0 and t % tc_sched[2] in tc_sched[:2]:
        # if np.random.rand() < 1./100:
            if t % tc_sched[2] == tc_sched[0]:
                # tok = ["left","right"][np.random.randint(2)]
                tok = "right"
            if t % tc_sched[2] == tc_sched[1]:
                tok = "cross"
            nvmnet.activity["tc"] = nvmnet.layers["tc"].coder.encode(tok) # maybe face appears

        ### show state and tick
        # if True:
        if t % 2 == 0 or nvmnet.at_exit():
        # if nvmnet.at_start() or nvmnet.at_exit():
            if nvmnet.at_start(): start_t.append(t)
            print('t = %d'%t)
            print(nvmnet.state_string(show_layers, show_tokens, show_corrosion, show_gates))
            # raw_input(".")
        if nvmnet.at_exit():
            break

        history.append(dict(nvmnet.activity))
        nvmnet.tick()
        
    ### raster plot
    A = np.zeros((sum([
        nvmnet.layers[name].size for sl in show_layers for name in sl]),
        len(history)))
    for h in range(len(history)):
        A[:,[h]] = np.concatenate([history[h][k] for sl in show_layers for k in sl],axis=0)
    
    xt = start_t
    xl = []
    for t in start_t:
        ops = []
        for op in ["opc","op1","op2"]:
            tok = nvmnet.layers[op].coder.decode(history[t][op])
            ops.append("" if tok in ["null","?"] else tok)
        xl.append("\n".join([str(t)]+ops))
    yt = np.array([history[0][k].shape[0] for sl in show_layers for k in sl])
    yt = yt.cumsum() - yt/2
    
    act = nvmnet.layers["gh"].activator
    plt.figure()
    plt.imshow(A, cmap='gray', vmin=act.off, vmax=act.on, aspect='auto')
    plt.xticks(xt, xl)
    plt.yticks(yt, [k for sl in show_layers for k in sl])
    plt.show()
    
