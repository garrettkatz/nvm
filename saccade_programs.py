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

aas_program = {"aas":"""

        set wm1 act
        set wm2 loop
loop:   set fef center
        set sc off
wait:   jmp tc
left:   set fef right
        jmp wm1
right:  set fef left
        jmp wm1
act:    set sc on
        jmp wm2
        exit
    
"""}

def make_fef(pad, activator, dim):
    act = activator(pad, dim*dim)
    fef_coder = Coder(act)
    fef_coder.encode("center")
    fef_coder.encode("left")
    fef_coder.encode("right")
    return Layer("fef", (dim,dim), act, fef_coder)

if __name__ == "__main__":
    
    # set up activator
    activator, learning_rule = logistic_activator, logistic_hebbian
    # activator, learning_rule = tanh_activator, tanh_hebbian

    # make network
    layer_size = 650
    pad = 0.001
    act = activator(pad, layer_size)
    
    device_names = ["wm1","wm2","tc","sc"]
    devices = {name: Layer(name, (layer_size,1), act, Coder(act))
        for name in device_names}
    devices["fef"] = make_fef(pad, activator, 32)

    # assemble and link programs
    nvmnet = NVMNet(layer_size, pad, activator, learning_rule, devices)
    for name, program in aas_program.items():
        nvmnet.assemble(program, name, verbose=1)
    nvmnet.link(verbose=1)

    # initialize layers
    nvmnet.activity["ip"] = nvmnet.layers["ip"].coder.encode(name) # program pointer
    nvmnet.activity["tc"] = nvmnet.layers["tc"].coder.encode("wait") # waiting for face

    raw_input("continue?")
        
    show_layers = [
        ["go", "gh","ip"] + ["op"+x for x in "c123"] + device_names,
    ]
    show_tokens = True
    show_corrosion = True
    show_gates = False

    history = []
    start_t = []
    for t in range(200):
    
        ### occassionally change tc
        if t > 0 and t % 100 == 0:
            tok = ["wait","left","right"][np.random.randint(3)]
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
        for op in ["opc","op1","op2","op3"]:
            tok = nvmnet.layers[op].coder.decode(history[t][op])
            ops.append("" if tok in ["null","?"] else tok)
        xl.append("\n".join([str(t)]+ops))
    yt = np.array([history[0][k].shape[0] for sl in show_layers for k in sl])
    yt = yt.cumsum() - yt/2
    
    plt.figure()
    plt.imshow(A, cmap='gray', vmin=act.off, vmax=act.on, aspect='auto')
    plt.xticks(xt, xl)
    plt.yticks(yt, [k for sl in show_layers for k in sl])
    plt.show()
    
