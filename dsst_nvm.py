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

dsst_programs = {
"sub":"""

    start:  sub test
            mov tc A
            jmp end
    test:   mov tc B
            ret
    end:    exit
""",
# "dsst":"""

#     # do one saccade (move and then hold)
#     # intended direction should be in premotor
#     sacc:   mov mc pm
#             mov mc hold
#             ret

#     # move to end of dsst area
#     # intended direction should be in premotor
#     tend:   cmp tc bound
#             jie back
#             mov mc left
#             mov mc hold

# """
}

def make_dsst_nvm(activator_label, tokens=[]):

    # set up activator
    if activator_label == "logistic":
        activator = logistic_activator
    if activator_label == "tanh":
        activator = tanh_activator
    learning_rule = hebbian

    # make network
    layer_shape = (1500,1)
    layer_size = layer_shape[0]*layer_shape[1]
    pad = 0.0001
    act = activator(pad, layer_size)
    
    devices = {
        "ol": Layer("ol", layer_shape, act, Coder(act)), # occipital lobe
        "tc": Layer("tc", layer_shape, act, Coder(act)), # temporal cortex
        "mc": Layer("mc", layer_shape, act, Coder(act)),} # motor cortex

    # assemble and link programs
    nvmnet = NVMNet(layer_shape, pad, activator, learning_rule, devices, gh_shape=(32,16))
    for name, program in dsst_programs.items():
        nvmnet.assemble(program, name, verbose=1)
    diff_count = nvmnet.link(verbose=2, tokens=tokens)

    return nvmnet, diff_count

if __name__ == "__main__":
    
    tokens = []
    diff_count = 10
    while diff_count > 5:
        nvmnet, diff_count = make_dsst_nvm("logistic", tokens=tokens)
    # nvmnet = make_nback_nvm("tanh")
    # raw_input("continue?")
    
    show_layers = [
        ["go", "gh","ip"] + ["op"+x for x in "c12"] +\
        ["mf","mb"] + ["sf","sb"] + ["co","ci"] +\
        nvmnet.devices.keys(),
    ]
    show_tokens = True
    show_corrosion = True
    show_gates = False

    nvmnet.load("sub", {})

    history = []
    start_t = []
    for t in range(100):
    
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
    
