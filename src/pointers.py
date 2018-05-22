import numpy as np
import matplotlib.pyplot as plt
from layer import Layer
from coder import Coder
from gate_map import make_nvm_gate_map
from activator import *
from learning_rules import *
from nvm_instruction_set import flash_instruction_set
from nvm_assembler import assemble
from nvm_linker import link
from nvm_net import NVMNet

pointer_programs = {
"poi":"""

        mov r1 A
        mem r1
        mov r2 X
        ref r2
        nxt
        mov r1 B
        mem r1
        drf r2
        rem r1
        exit

""",
}


def make_pointer_nvm(activator_label, tokens=[]):

    # set up activator
    if activator_label == "logistic":
        activator = logistic_activator
    if activator_label == "tanh":
        activator = tanh_activator
    learning_rule = hebbian

    # make network
    layer_shape = (1200,1)
    layer_size = layer_shape[0]*layer_shape[1]
    pad = 0.0001
    act = activator(pad, layer_size)
    
    devices = {
        "r1": Layer("r1", layer_shape, act, Coder(act)), # register 1
        "r2": Layer("r2", layer_shape, act, Coder(act)), # register 2
        }

    # assemble and link programs
    shapes = {}
    nvmnet = NVMNet(layer_shape, pad, activator, learning_rule, devices, shapes=shapes)
    for name, program in pointer_programs.items():
        nvmnet.assemble(program, name, verbose=1)
    diff_count = nvmnet.link(verbose=2, tokens=tokens)

    return nvmnet, diff_count

if __name__ == "__main__":
    
    diff_count = 10
    while diff_count > 5:
        # nvmnet, diff_count = make_pointer_nvm("logistic")
        nvmnet, diff_count = make_pointer_nvm("tanh")
        break
    # raw_input("continue?")
    
    show_layers = [
        ["go", "gh","ip"] + ["op"+x for x in "c12"] +\
        ["mf","mb"] + ["co","ci"] +\
        nvmnet.devices.keys(),
    ]
    show_tokens = True
    show_corrosion = False
    show_gates = False

    nvmnet.load("poi", {})

    history = []
    start_t = []
    for t in range(100):
    
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
    plt.tight_layout()
    plt.show()
    
