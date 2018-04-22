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

            sub foo
            exit

    foo:    mov tc A
            sub bar
            mov tc B
            sub bar
            mov tc C
            ret

    bar:    mov tc D
            ret
    
""",
# "dsst":"""

#             sub lookup
#             exit

#     # look up current symbol in key
#     # intended symbol should be in occipital lobe
#             # put current symbol in temporal cortex
#     lookup: mov tc ol
#             # look left and up to find key
#             mov pm left
#             sub tend
#             mov pm up
#             sub tend
#             # check current symbol against key
#     check:  cmp tc ol
#             jie match
#             # if no match, step through key
#             mov pm right
#             sub step
#             jmp check
#             # if match, put matching digit below in temporal cortex
#     match:  mov pm down
#             sub step
#             mov tc ol
#             # lookup done
#             ret

#     # do one step (move and then hold)
#     # intended direction should be in premotor
#     step:   mov mc pm
#             mov mc hold
#             ret

#     # move to boundary of dsst area
#     # intended direction should be in premotor
#             # compare current view with boundary (+)
#     tend:   cmp ol +
#             jie back
#             # if not at boundary yet, keep stepping
#             sub step
#             jmp tend
#             # if at boundary, take one step back
#     back:   cmp pm left
#             jie left
#             cmp pm right
#             jie right
#             cmp pm down
#             jie down
#             # one step in opposite direction
#     up:     mov pm down
#             jmp done
#     down:   mov pm up
#             jmp done
#     right:  mov pm left
#             jmp done
#     left:   mov pm right
#     done:   sub step
#             ret

#             exit
# """
}

def make_dsst_grid(letters, rows, cols):

    grid = [["+" for c in range(cols+2)] for r in range(2*rows+2)]
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c < len(letters):
                grid[2*r+1][c+1] = letters[c]
                grid[2*r+2][c+1] = str(c)
            else:
                grid[2*r+1][c+1] = letters[np.random.randint(len(letters))]
                grid[2*r+2][c+1] = "_"
    return grid

def make_dsst_nvm(activator_label, tokens=[]):

    # set up activator
    if activator_label == "logistic":
        activator = logistic_activator
    if activator_label == "tanh":
        activator = tanh_activator
    learning_rule = hebbian

    # make network
    layer_shape = (1600,1)
    layer_size = layer_shape[0]*layer_shape[1]
    pad = 0.0001
    act = activator(pad, layer_size)
    
    devices = {
        "ol": Layer("ol", layer_shape, act, Coder(act)), # occipital lobe
        "tc": Layer("tc", layer_shape, act, Coder(act)), # temporal cortex
        "pm": Layer("pm", layer_shape, act, Coder(act)), # premotor cortex
        "mc": Layer("mc", layer_shape, act, Coder(act)),} # motor cortex

    # assemble and link programs
    shapes = {"gh":(32,16), "s":(32,16)}
    nvmnet = NVMNet(layer_shape, pad, activator, learning_rule, devices, shapes=shapes)
    for name, program in dsst_programs.items():
        nvmnet.assemble(program, name, verbose=1)
    diff_count = nvmnet.link(verbose=2, tokens=tokens)

    return nvmnet, diff_count

if __name__ == "__main__":
    
    letters = list('abcd')
    grid = make_dsst_grid(letters, 2, len(letters))
    # print("\n".join([" ".join(g) for g in grid]))
    # raw_input('.')

    # _ is blank space
    # + is grid boundary
    tokens = ["up","left","down","right","_","+"] + letters + map(str,range(len(letters)))
    diff_count = 100
    while diff_count > 10:
        nvmnet, diff_count = make_dsst_nvm("logistic", tokens=tokens)
        break
    # nvmnet = make_nback_nvm("tanh")
    # raw_input("continue?")
    
    show_layers = [
        ["go", "gh","ip"] + ["op"+x for x in "c12"] +\
        ["mf","mb"] + ["sf","sb"] + ["co","ci"] +\
        nvmnet.devices.keys(),
    ]
    show_tokens = True
    show_corrosion = False
    show_gates = False

    grid_row, grid_col = 3, 2
    moved = False
    grids = [(grid_row, grid_col, [list(g) for g in grid])]
    
    nvmnet.load("sub", {})
    # nvmnet.load("dsst", {
    #     "mc": "hold"
    # })

    history = []
    start_t = []
    trace = []
    for t in range(300):
    
        # Action
        motion = nvmnet.layers['mc'].coder.decode(nvmnet.activity['mc'])
        if not motion == 'hold' and not moved:
            if motion == 'left' and grid_col > 0: grid_col -= 1
            if motion == 'right' and grid_col < len(grid[0])-1: grid_col += 1
            if motion == 'up' and grid_row > 0: grid_row -= 1
            if motion == 'down' and grid_row < len(grid)-1: grid_row += 1
            moved = True
            grids.append((grid_row, grid_col, [list(g) for g in grid]))
        if motion == 'hold': moved = False

        # Perception
        nvmnet.activity['ol'] = nvmnet.layers['ol'].coder.encode(grid[grid_row][grid_col])

        ### show state and tick
        # if True:
        # if t % 2 == 0 or nvmnet.at_exit():
        if nvmnet.at_start() or nvmnet.at_exit():
            if nvmnet.at_start():
                start_t.append(t)
                trace_st = nvmnet.state_string(
                    [['ip','opc','op1','op2','sf','sb']], True, False, False)
                trace.append(trace_st[:-1]) # w/o newline
            print('t = %d'%t)
            print(nvmnet.state_string(show_layers, show_tokens, show_corrosion, show_gates))
            # raw_input(".")
        if nvmnet.at_exit():
            break
        nvmnet.tick()

        history.append(dict(nvmnet.activity))

    ### Grid history
    for i, (r, c, g) in enumerate(grids):
        print("*** GRID %d ***"%i)
        print("At %d, %d"%(r,c))
        print("\n".join([" ".join(g) for g in grid]))
        # raw_input('...')
    
    print("*** execution trace ***")
    for t in trace:
        print(t)

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
    
