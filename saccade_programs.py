import numpy as np
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

if __name__ == "__main__":
    
    # set up activator
    activator, learning_rule = logistic_activator, logistic_hebbian
    # activator, learning_rule = tanh_activator, tanh_hebbian

    # make network
    layer_size = 650
    pad = 0.001
    act = activator(pad, layer_size)
    
    device_names = ["wm1","wm2","fef","tc","sc"]
    devices = {name: Layer(name, layer_size, act, Coder(act))
        for name in device_names}

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

    for t in range(500):
    
        ### occassionally change tc
        if t > 0 and t % 100 == 0:
            tok = ["wait","left","right"][np.random.randint(3)]
            nvmnet.activity["tc"] = nvmnet.layers["tc"].coder.encode(tok) # maybe face appears

        ### show state and tick
        # if True:
        # if t % 2 == 0 or nvmnet.at_exit():
        if nvmnet.at_start() or nvmnet.at_exit():
            print('t = %d'%t)
            print(nvmnet.state_string(show_layers, show_tokens, show_corrosion, show_gates))
            raw_input(".")
        if nvmnet.at_exit():
            break
        nvmnet.tick()
