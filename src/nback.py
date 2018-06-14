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

nback_programs = {
# "mem":"""

#     start:  mov tc A
#             mem tc
#             nxt
#             mov tc B
#             prv
#             rem tc
#     overw:  mov tc C
#             mem tc
#             nxt
#             mov tc B
#             prv
#             rem tc
#     end:    exit
# """,

# "cmp":"""

#     start:  mov tc A
#             mov mc B
#             cmp tc mc
#             jie eq
#             mov mc no
#             jmp end
#     eq:     mov mc yes
#     end:    exit

# """,

# "cmpv":"""

#     start:  mov tc A
#             cmp tc A
#             jie eq
#             mov mc no
#             jmp end
#     eq:     mov mc yes
#     end:    exit

# """,

"nback":"""

            jmp start{n}

    start2: mov tc ol
            mem tc
            nxt
            mov mc no
            mov mc hold
    start1: mov tc ol
            mem tc
            nxt
            mov mc no
            mov mc hold

    repeat: mov tc ol
            mem tc
            jmp back{n}

    back2:  prv
    back1:  prv
            rem tc
            cmp tc ol
            jie match

            mov mc no
            jmp hold
    match:  mov mc yes
    hold:   mov mc hold
            jmp forw{n}
            
    forw2:  nxt
    forw1:  nxt
            nxt
            jmp repeat

            exit
    
""".format(n=1)
}

def make_nback_nvm(activator_label, tokens=[]):

    # set up activator
    if activator_label == "logistic":
        activator = logistic_activator
    if activator_label == "tanh":
        activator = tanh_activator
    # learning_rule = hebbian
    learning_rule = rehebbian

    # make network
    layer_shape = (1024,1)
    layer_size = layer_shape[0]*layer_shape[1]
    pad = 0.0001
    act = activator(pad, layer_size)
    
    devices = {
        "ol": Layer("ol", layer_shape, act, Coder(act)), # occipital lobe
        "tc": Layer("tc", layer_shape, act, Coder(act)), # temporal cortex
        "mc": Layer("mc", layer_shape, act, Coder(act)),} # motor cortex

    # assemble and link programs
    shapes = {"gh":(32,16)}
    nvmnet = NVMNet(layer_shape, pad, activator, learning_rule, devices, shapes=shapes)
    for name, program in nback_programs.items():
        nvmnet.assemble(program, name, verbose=1, other_tokens=tokens)
    # diff_count = nvmnet.link(verbose=2, tokens=tokens, orthogonal=True)

    return nvmnet, diff_count

if __name__ == "__main__":
    
    letters = list('abcd')
    diff_count = 10
    while diff_count > 5:
        nvmnet, diff_count = make_nback_nvm("logistic", tokens=letters)
        # nvmnet, diff_count = make_nback_nvm("tanh", tokens=letters)
        break
    # nvmnet = make_nback_nvm("tanh")
    # raw_input("continue?")
    
    show_layers = [
        ["go", "gh","ip"] + ["op"+x for x in "c12"] +\
        ["mf","mb"] + ["co","ci"] +\
        nvmnet.devices.keys(),
    ]
    show_tokens = True
    show_corrosion = False
    show_gates = False

    # prompts = np.array(letters)[np.random.randint(letters.size,size=10)]
    prompts = ['a','a','b','a','a']
    prompt_index = 0
    # print(prompts)
    # raw_input("continue?")

    nvmnet.load("nback", {
        "mc": "hold",
        "ol": prompts[0],
    })
    # nvmnet.load("mem", {})
    # nvmnet.load("cmp", {})
    # nvmnet.load("cmpv", {})

    history = []
    start_t = []
    responded = False
    responses = []
    for t in range(600):
    
        response = nvmnet.layers['mc'].coder.decode(nvmnet.activity['mc'])
        if response == 'hold': responded = False
        elif not responded:
            responded = True
            responses.append(response)
            prompt_index += 1
            if prompt_index >= len(prompts): break
            p = nvmnet.layers['ol'].coder.encode(prompts[prompt_index])
            nvmnet.activity['ol'] = p

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

    print('******** prompts, responses *********')
    print(prompts)
    print(responses)
    
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
    
