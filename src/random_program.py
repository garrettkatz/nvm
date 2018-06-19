"""
Experiment: want to assess program capacity.
    variables:
    how many lines/tokens in the program that have to be stored
    how well vm performs (leading match count out of total trial steps, or binary success/fail)
    how big the vm is (layer size/total # units and connections)
    orthogonal vs not.
    
    plot:
    different markers for orthogonal vs not
    different grayscales for vm layer sizes
    match count/total steps (ratio or line connected? fixed max steps?) vs program size
    
    or:
    
    different marker/color for orthogonal vs not
    different marker/color for success vs not
    program size vs network size
"""
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
from matplotlib.lines import Line2D
import multiprocessing as mp
import itertools as it
from nvm import make_default_nvm
from preprocessing import preprocess
from refvm import RefVM

hr_opcodes = np.array([
    "mov","jmp",
    "cmp","jie",
    "sub",
    "mem","nxt","prv","ref",
    # "rem","drf",
    ], dtype=object)

def generate_random_program(register_names, num_tokens, num_subroutines, num_lines):

    # Initialize tokens
    tokens = ['tok'+str(t) for t in range(num_tokens)]

    # Initialize line indices of subroutine code
    subroutine_partition = np.random.choice(
        np.arange(2,num_lines), # leave >= 2 for sub and exit in main
        num_subroutines, replace=False)
    subroutine_partition.sort()
    subroutine_partition = np.insert(subroutine_partition, 0, 0)
    subroutine_partition = np.append(subroutine_partition, num_lines)

    # Randomize opcodes on each line
    lines = np.random.choice(hr_opcodes, num_lines)

    # Make sure main code ends in exit and subroutines end in ret
    lines[subroutine_partition[1]-1] = 'exit'
    for sp in subroutine_partition[2:]: lines[sp-1] = 'ret'

    # Make sure at least one subroutine is called from main
    lines[np.random.randint(subroutine_partition[1]-1)] = 'sub'

    # Make labels for subroutines
    labels = {"lab%d"%l: l for l in subroutine_partition[1:-1]}

    # Add operands including labels for jumps
    for l in range(len(lines)):

        if lines[l] in ["mov","cmp"]:
            lines[l] += " %s %s"%(
                np.random.choice(register_names),
                np.random.choice(register_names + tokens))
        if lines[l] in ["jie","jmp"]:
            p = np.argmax(subroutine_partition > l)
            label_line = np.random.randint( # jmp within subroutine
                subroutine_partition[p-1], subroutine_partition[p])
            label = "lab%d"%label_line
            lines[l] += " %s"%label
            labels[label] = label_line # save new label
        if lines[l] == "sub":
            label = "lab%d"%np.random.choice(subroutine_partition[1:-1])
            lines[l] += " %s"%label
        if lines[l] in ["mem","rem","ref","drf"]:
            lines[l] += " %s"%np.random.choice(register_names)

    # Prepend labels to program lines
    for l in range(len(lines)):
        lines[l] = " "*8 + lines[l]
    for label, l in labels.items():
        lines[l] = label + ":" + lines[l][len(label)+1:]
    
    program = "\n".join(lines)
    initial_activity = {r: np.random.choice(tokens) for r in register_names}

    return program, tokens, initial_activity

def run_random_program(vm, program, tokens, initial_activity, max_steps, verbose=0):

    # Load and run
    vm.assemble(program, "rand", verbose=0, other_tokens=tokens)
    vm.load("rand", initial_activity)

    trace = []
    for t in range(max_steps):
        if vm.at_exit(): break
        if verbose > 0: print("step %d: %s"%(t, vm.state_string()))
        vm.step(verbose=0, max_ticks=20)
        trace.append(vm.decode_state(layer_names=[
            "opc","op1","op2"] + vm.register_names))
    
    return trace

def match_trial(
    register_names,
    num_tokens,
    num_subroutines,
    num_lines,
    layer_shape,
    orthogonal,
    max_steps,
    verbose=0):

    # Generate program
    program, tokens, initial_activity = generate_random_program(
        register_names, num_tokens, num_subroutines, num_lines)

    # NVM run
    nvm = make_default_nvm(register_names, layer_shape=layer_shape, orthogonal=orthogonal)
    nvm_trace = run_random_program(nvm,
        program, tokens, initial_activity, max_steps=max_steps, verbose=0)

    # RVM run
    rvm = RefVM(register_names)
    rvm_trace = run_random_program(rvm,
        program, tokens, initial_activity, max_steps=max_steps, verbose=0)

    for t in range(len(rvm_trace)):
        if t >= len(nvm_trace): break
        if nvm_trace[t] != rvm_trace[t]: break
    
    leading_match_count = t+1
    trial_step_count = len(rvm_trace)
    
    return program, leading_match_count, trial_step_count, nvm_trace, rvm_trace

def match_trial_caller(args):
    return match_trial(*args)

def run_match_trial_pool(args_list, num_procs=0):

    if num_procs > 0: # multiprocessed

        num_procs = min(num_procs, mp.cpu_count())
        print("using %d processes..."%num_procs)
        pool = mp.Pool(processes=num_procs)
        results = pool.map(match_trial_caller, args_list)
        pool.close()
        pool.join()

    else: # serial

        print("single processing...")
        results = []
        for a, args in enumerate(args_list):
            print('trial %d of %d'%(a, len(args_list)))
            results.append(match_trial_caller(args))

    return results

def plot_trial_complexities(args_list, results):
    for a in range(len(args_list)):
        register_names, program = args_list[a][0], results[a][0]
        lines, labels = preprocess(program, register_names)
        num_tokens = len(set([tok for line in lines for tok in line] + labels.keys()))
        pt.scatter(len(lines), num_tokens, c='k')
    pt.xlabel('Program length (lines)')
    pt.ylabel('Number of unique tokens')
    pt.show()

"""
color or bargroup by network size (too striated)
mark by orth
plot: relate success to program complexity
"""

def plot_match_trials(args_list, results, layer_shape_colors):

    for a in range(len(args_list)):
        register_names, program = args_list[a][0], results[a][0]
        num_lines, layer_shape, orthogonal = args_list[a][3:6]
        leading_match_count, trial_step_count = results[a][1:3]
        lines, labels = preprocess(program, register_names)
        num_tokens = len(set([tok for line in lines for tok in line] + labels.keys()))
        
        success = (leading_match_count == trial_step_count)
        # if  orthogonal and  success:
        #     pt.scatter(num_tokens, layer_shape[0], marker='o',facecolors='none',edgecolors='k')
        if  orthogonal and ~success:
            pt.scatter(num_tokens, layer_shape[0], marker='o',facecolors='none',edgecolors='gray')
        if ~orthogonal and  success:
            pt.scatter(num_tokens, layer_shape[0], marker='+',c='k')
        # if ~orthogonal and ~success:
        #     pt.scatter(num_tokens, layer_shape[0], marker='+',c='gray')
        # if  orthogonal and  success: kwa = dict(marker='o',c='k')
        # if  orthogonal and ~success: kwa = dict(marker='o')
        # if ~orthogonal and  success: kwa = dict(marker='+',c='k')
        # if ~orthogonal and ~success: kwa = dict(marker='+',c='k')
        # # if orthogonal and success: pt.scatter([0],[0],marker='o',c='k')
        # if orthogonal and success:
        #     print(num_tokens, layer_shape[0])
        #     pt.scatter(num_tokens, layer_shape[0], **kwa)

    pt.legend([
        Line2D([0],[0], marker='o', color='none',markeredgecolor='k', linestyle='none'),
        Line2D([0],[0], marker='+', color='k', linestyle='none'),
        Line2D([0],[0], marker='s', color='k', linestyle='none'),
        Line2D([0],[0], marker='s', color='gray', linestyle='none'),
    ],["Orth","Rand","Success","Fail"])
    pt.xlabel("# unique program tokens")
    pt.ylabel("Layer size") 
    pt.show()


# def plot_match_trials(args_list, results, layer_shape_colors):

#     for a in range(len(args_list)):
#         register_names, program = args_list[a][0], results[a][0]
#         num_lines, layer_shape, orthogonal = args_list[a][3:6]
#         leading_match_count, trial_step_count = results[a][1:3]
#         lines, labels = preprocess(program, register_names)
#         num_tokens = len(set([tok for line in lines for tok in line] + labels.keys()))
        
#         c = layer_shape_colors[layer_shape]
#         color = (c,c,c)
#         if orthogonal:
#             pt.scatter(num_tokens, float(leading_match_count)/trial_step_count,
#                 marker='o',color='none',edgecolor=color)
#         else:
#             pt.scatter(num_tokens, float(leading_match_count)/trial_step_count,
#                 marker='+',color=color)

#     NN = np.sort(np.array([s[0] for s in layer_shape_colors]))
#     pt.legend([
#         Line2D([0],[0], marker='o', c='none', markeredgecolor='k'),
#         Line2D([0],[0], marker='+', c='k', linestyle='none')] + \
#         [Line2D([0],[0], marker='s', linestyle='none', c=(layer_shape_colors[(N,1)],)*3) for N in NN],
#         ["Orth","Rand"] + ["N = %d"%N for N in NN])
#     pt.xlabel("# unique program tokens")
#     pt.ylabel("Leading portion of correct instruction executions") 
#     pt.show()

# def plot_match_trials(args_list, results, layer_shape_colors):

#     for a in range(len(args_list)):
#         register_names, program = args_list[a][0], results[a][0]
#         num_lines, layer_shape, orthogonal = args_list[a][3:6]
#         leading_match_count, trial_step_count = results[a][1:3]
#         lines, labels = preprocess(program, register_names)
#         num_tokens = len(set([tok for line in lines for tok in line] + labels.keys()))
        
#         c = 0. if leading_match_count == trial_step_count else 0.5
#         color = (c,c,c)
#         if orthogonal:
#             pt.scatter(layer_shape[0]*layer_shape[1], num_tokens,
#                 marker='o',color='none',edgecolor=color)
#         else:
#             pt.scatter(layer_shape[0]*layer_shape[1], num_tokens,
#                 marker='+',color=color)

#     pt.show()

if __name__ == "__main__":

    num_registers = 3
    register_names = ["r%d"%r for r in range(num_registers)]
    num_tokens = 2
    max_steps = 50
    verbose = 0
    program_sizes = [ # num_subroutines, num_lines
        (1, 8),
        (1, 16),
        (2, 24),
        (3, 36),
        (3, 48),
    ]
    layer_shapes = [
        (16, 1),
        (32, 1),
        (64, 1),
        (128,1),
        (256,1),
        # (512,1),
    ]
    layer_shape_colors = {
        s: np.arange(0, len(layer_shapes), dtype=float)[::-1][i]/len(layer_shapes)
        for i,s in enumerate(layer_shapes)}
    # layer_shape_colors = {
    #     (16, 1): .8,
    #     (32, 1): .6,
    #     (64, 1): .4,
    #     (128,1): .2,
    #     (256,1): .0,
    #     # (512,1): .0,
    # }
    
    args_list = [(
        register_names, num_tokens, num_subroutines, num_lines,
        layer_shape, orthogonal, max_steps, verbose)
        for (orthogonal, (num_subroutines, num_lines), layer_shape) in it.product(
            [False, True], program_sizes, layer_shapes)]

    # results = run_match_trial_pool(args_list, num_procs=0)
    # with open('tmp.pkl','w') as f: pk.dump(results, f)

    with open('tmp.pkl','r') as f: results = pk.load(f)
    
    for a, (_, leading_match_count, trial_step_count, _, _) in enumerate(results):
        print("|prog|=%d, N=%d, %s: %d of %d"%(
            args_list[a][3], args_list[a][4][0],
            "orth" if args_list[a][5] else "rand",
            leading_match_count, trial_step_count))

    # plot_trial_complexities(args_list, results)
    plot_match_trials(args_list, results, layer_shape_colors)

    # num_registers = 3
    # register_names = ["r%d"%r for r in range(num_registers)]
    # num_tokens = 1
    # num_subroutines = 2
    # num_lines = 10
    # layer_shape = (32,32)
    # orthogonal=True
    # max_steps = 5

    # program, leading_match_count, trial_step_count, nvm_trace, rvm_trace = match_trial(
    #     register_names, num_tokens, num_subroutines, num_lines,
    #     layer_shape, orthogonal, max_steps, verbose=0)

    # for t in nvm_trace: print(t)
    # print(program)
    # for t in rvm_trace: print(t)
    # print("matched leading %d of %d"%(leading_match_count, trial_step_count))
