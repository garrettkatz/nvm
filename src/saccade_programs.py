import numpy as np
import matplotlib.pyplot as plt
from layer import Layer
from coder import Coder
from gate_map import make_nvm_gate_map
from activator import *
from learning_rules import *
from nvm_instruction_set import flash_instruction_set
from nvm_assembler import assemble
# from nvm_linker import link
from nvm_net import NVMNet

aas_program = {"aas":"""

loop:   mov fef center
        mov sc on
        mov sc off
wait:   cmp tc cross
        jie cross
        jmp wait
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

def make_fef(pad, activator, rows, columns):
    dim = min(rows, columns)

    act = activator(pad, rows*columns)
    fef_coder = Coder(act)

    Y, X = np.mgrid[:rows,:columns] # transpose for bitmap
    R = .1
    # # Cts
    # center = act.off + (act.on-act.off)*np.exp(-((X-.5*columns)**2 + (Y-.5*rows)**2)/(R*dim)**2)
    # left = act.off + (act.on-act.off)*np.exp(-((X-.0*columns)**2 + (Y-.5*rows)**2)/(R*dim)**2)
    # right = act.off + (act.on-act.off)*np.exp(-((X-1.*columns)**2 + (Y-.5*rows)**2)/(R*dim)**2)
    # Binary
    center = act.off + (act.on-act.off)*((X-.5*columns)**2 + (Y-.5*rows)**2 < (R*dim)**2)
    left = act.off + (act.on-act.off)*((X-.0*columns)**2 + (Y-.5*rows)**2 < (R*dim)**2)
    right = act.off + (act.on-act.off)*((X-1.*columns)**2 + (Y-.5*rows)**2 < (R*dim)**2)

    fef_coder.encode("center", center.reshape((rows*columns,1)))
    fef_coder.encode("left", left.reshape((rows*columns,1)))
    fef_coder.encode("right", right.reshape((rows*columns,1)))
    return Layer("fef", (rows,columns), act, fef_coder)

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
    # learning_rule = hebbian
    learning_rule = rehebbian

    # make network
    layer_shape = (32,32)
    layer_size = layer_shape[0]*layer_shape[1]
    pad = 0.0001
    act = activator(pad, layer_size)
    
    devices = {
        "tc": Layer("tc", layer_shape, act, Coder(act)),
        "fef": make_fef(pad, activator, 68, 96),
        "sc": make_sc(pad, activator, 5, 5)}

    # assemble and link programs
    shapes = {"gh": (32,16)}
    nvmnet = NVMNet(layer_shape, pad, activator, learning_rule, devices, shapes=shapes)
    for name, program in aas_program.items():
        nvmnet.assemble(program, name, verbose=1)
    # nvmnet.link(verbose=2)

    # initialize layers
    nvmnet.activity["ip"] = nvmnet.layers["ip"].coder.encode(name) # program pointer
    nvmnet.activity["tc"] = nvmnet.layers["tc"].coder.encode("cross") # waiting for face

    return nvmnet

if __name__ == "__main__":
    
    nvmnet = make_saccade_nvm("logistic")
    raw_input("continue?")
    
    show_layers = [
        ["go", "gh","ip"] + ["op"+x for x in "c12"] + nvmnet.devices.keys(),
    ]
    show_tokens = True
    show_corrosion = True
    show_gates = False

    history = []
    start_t = []
    tc_sched = [60, 110, 120]
    for t in range(tc_sched[2]*3):
    
        ### occassionally change tc
        if t > 0 and t % tc_sched[2] in tc_sched[:2]:
        # if np.random.rand() < 1./100:
            if t % tc_sched[2] == tc_sched[0]:
                tok = ["left","right"][np.random.randint(2)]
            if t % tc_sched[2] == tc_sched[1]:
                tok = "cross"
            nvmnet.activity["tc"] = nvmnet.layers["tc"].coder.encode(tok) # maybe face appears

        ### show state and tick
        # if True:
        # if t % 2 == 0 or nvmnet.at_exit():
        if nvmnet.at_start() or nvmnet.at_exit():
            if nvmnet.at_start(): start_t.append(t)
            print('t = %d'%t)
            print(nvmnet.state_string(show_layers, show_tokens, show_corrosion, show_gates))
            # raw_input(".")
        if nvmnet.at_exit():
            break
        nvmnet.tick()

        history.append(dict(nvmnet.activity))
        
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
    
