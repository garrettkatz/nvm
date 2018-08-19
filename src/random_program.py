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

    # Randomize opcodes on each line
    lines = np.random.choice(hr_opcodes, num_lines)

    # Initialize line indices of subroutine code
    subroutine_partition = np.random.choice(
        np.arange(2,num_lines), # leave >= 2 for sub and exit in main
        num_subroutines, replace=False)
    subroutine_partition.sort()
    subroutine_partition = np.insert(subroutine_partition, 0, 0)
    subroutine_partition = np.append(subroutine_partition, num_lines)

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
        lines[l] = " "*10 + lines[l]
    for label, l in labels.items():
        lines[l] = label + ":" + lines[l][len(label)+1:]
    
    program = "\n".join(lines)
    initial_activity = {r: np.random.choice(tokens) for r in register_names}
    tokens += labels.keys()

    return program, tokens, initial_activity

def print_random_program(register_names, num_tokens, num_subroutines, num_lines):
    program, _, _ = generate_random_program(register_names, num_tokens, num_subroutines, num_lines)
    lines, labels = preprocess(program, register_names)
    distinct_tokens = set([tok for line in lines for tok in line] + labels.keys())
    print(program)
    print("")
    print(list(distinct_tokens))
    print("%d distinct tokens"%len(distinct_tokens))

def run_program(vm, program, name, initial_activity, max_steps, verbose=0):

    # Load and run
    vm.assemble(program, name, verbose=0, other_tokens=tokens)
    vm.load(name, initial_activity)

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
    program_load,
    layer_shape,
    shapes,
    orthogonal,
    max_steps,
    verbose=0):

    # Make vms and assemble
    all_tokens = []
    for _, tokens, _ in program_load:
        all_tokens += tokens
    nvm = make_default_nvm(
        register_names, layer_shape=layer_shape, orthogonal=orthogonal,
        # shapes = {"ip": (max(
        #     layer_shape[0], 2**int(np.ceil(np.log2(num_lines+1)))),1) },
        tokens = all_tokens)
    rvm = RefVM(register_names)

    leading_match_counts, trial_step_counts, nvm_traces, rvm_traces = [], [], [], []
    for p, (program, tokens, initial_activity) in program_load:
    
        name = "rand%d"%p
        nvm_trace = run_program(nvm,
            program, name, initial_activity, max_steps=max_steps, verbose=0)
        rvm_trace = run_program(rvm,
            program, name, initial_activity, max_steps=max_steps, verbose=0)
    
        for t in range(len(rvm_trace)):
            if t >= len(nvm_trace): break
            if nvm_trace[t] != rvm_trace[t]: break
        
        leading_match_count = t+1
        trial_step_count = len(rvm_trace)
        leading_match_counts.append(leading_match_count)
        trial_step_counts.append(trial_step_count)
        nvm_traces.append(nvm_trace)
        rvm_traces.append(rvm_trace)
    
    return leading_match_counts, trial_step_counts, nvm_traces, rvm_traces

def match_trial_caller(args):
    try:
        return match_trial(*args)
    except Exception as e:
        print(e.message)
        return None

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

def plot_trial_complexities(register_names, program_loads):
    line_counts, token_counts = [], []
    for program_load in program_loads:
        tokens = []
        num_lines = 0
        for program, _, _ in program_load:
            lines, labels = preprocess(program, register_names)
            tokens += [tok for line in lines for tok in line] + labels.keys()
            num_lines += len(lines)
        num_tokens = len(set(tokens))
        line_counts.append(num_lines)
        token_counts.append(num_tokens)
    lc = np.array(line_counts)
    tc = np.array(token_counts)

    pt.figure(figsize=(6.5,2))
    # pt.scatter(lc, tc, c='none', edgecolor='k')
    pt.scatter(lc + 4*np.random.random_sample(*lc.shape), tc, c='k', marker='+')
    pt.xlabel('Program length (lines)')
    pt.ylabel('Distinct token count')
    pt.legend(["Tokens vs. lines"])
    pt.tight_layout()
    
    print(np.mean((lc - lc.mean())*(tc - tc.mean())/(lc.std()*tc.std())))    
    R = np.corrcoef(lc, tc)
    print("Pearson's R:")
    print(R)

    tc = tc[lc < 200]
    lc = lc[lc < 200]       
    print(np.mean((lc - lc.mean())*(tc - tc.mean())/(lc.std()*tc.std())))    
    R = np.corrcoef(lc, tc)
    print("Pearson's R (<200 lines):")
    print(R)
    pt.show()

def plot_match_trials(args_list, results, layer_shape_colors):
    sizes = {k: set() for k in [False, True]}
    line_counts = {k: set() for k in [False, True]}
    successes = {}
    failures = {}
    for a in range(len(args_list)):
        if results[a] is None: continue
        register_names, program = args_list[a][0], results[a][0]
        num_lines, layer_shape, orthogonal = args_list[a][3:6]
        leading_match_count, trial_step_count = results[a][1:3]
        lines, labels = preprocess(program, register_names)
        num_tokens = len(set([tok for line in lines for tok in line] + labels.keys()))
        success = (leading_match_count == trial_step_count)
        
        sizes[orthogonal].add(layer_shape[0])
        line_counts[orthogonal].add(num_lines)
        k = (layer_shape[0], num_lines, orthogonal)
        if success: successes[k] = successes.get(k, 0) + 1
        else: failures[k] = failures.get(k, 0) + 1

    sizes = {k: np.sort(list(sizes[k])) for k in [False, True]}
    line_counts = {k: np.sort(list(line_counts[k])) for k in [False, True]}
    y = {}
    y_orth = {}
    for orth in [False, True]:
        for size in sizes[orth]:
            if orth: y_orth[size] = []
            else: y[size] = []
            for count in line_counts[orth]:
                k = (size, count, orth)
                if orth:
                    y_orth[size].append(
                        float(successes.get(k,0))/(successes.get(k,0)+failures.get(k,0)))
                else:
                    y[size].append(
                        float(successes.get(k,0))/(successes.get(k,0)+failures.get(k,0)))

    xmx = max(line_counts[True].max(), line_counts[False].max())
    pt.figure(figsize=(7,5))
    pt.subplot(2,1,1)
    print("rand")
    h = []
    shades = np.linspace(0,0.5,len(sizes[False]))[::-1]
    for s,size in enumerate(sizes[False]):
        print(size, ["%d:%.2f"%tup for tup in zip(line_counts[False], y[size])])
        h.append(pt.plot(line_counts[False], y[size], '-o', c=3*[shades[s]])[0])
    pt.legend(h, ["N=%d"%s for s in sizes[False]])
    pt.ylabel("Success rate")
    pt.xlim([0, xmx])
    pt.title("Random Encodings")
    pt.subplot(2,1,2)
    print("orth")
    h = []
    shades = np.linspace(0,0.5,len(sizes[True]))[::-1]
    for s,size in enumerate(sizes[True]):
        print(size, ["%d:%.2f"%tup for tup in zip(line_counts[True], y_orth[size])])
        h.append(pt.plot(line_counts[True], y_orth[size], '-o', c=3*[shades[s]])[0])
    pt.legend(h, ["N=%d"%s for s in sizes[True]])
    pt.ylabel("Success rate")
    pt.xlabel("Line counts")
    pt.xlim([0, xmx])
    pt.title("Orthogonal Encodings")
    pt.tight_layout()
    pt.show()

    # To assess whether relationship between prog size and required nvm size linear, log, poly, etc:
    # for each line count, find smallest nvm size with perfect success rate.
    # then plot that line count vs required nvm size.
    rates = [.5, .8, .95, 1.]
    shades = [.7, .6, .1, 0] #np.linspace(0, .5, len(rates))[::-1]
    h = []
    pt.figure(figsize=(7,3.5))
    for orth in [False, True]:
        mk = 's' if orth else 'o'
        for r,rate in enumerate(rates):
            x, y = [], []
            for count in line_counts[orth]:
                x.append(count)
                best_size = np.inf
                for size in sizes[orth]:
                    k = (size, count, orth)
                    success_count = float(successes.get(k,0))
                    total_count = successes.get(k,0)+failures.get(k,0)
                    print(k, success_count, total_count)
                    if total_count == 0: continue
                    success_rate = success_count/total_count
                    # if success_rate == 1.0 and size < best_size: best_size = size
                    if success_rate >= rate and size < best_size: best_size = size
                y.append(best_size)
            h.append(pt.plot(x, y, '-', marker=mk, color=3*[shades[r]],markerfacecolor='none')[0])
    # fit logarithm: C log2(256) = 256 -> C = 256/log2(256) = 256/8 = 32
    # fit sqrt: C sqrt(256) = 256 -> C = 256/sqrt(256) = 16
    pt.plot(np.arange(1,1000), 32*np.log2(np.arange(1,1000)), '--k')
    # pt.plot(np.arange(1,1000), 16*np.sqrt(np.arange(1,1000)), '--k')
    pt.legend([
        Line2D([0],[0], marker='o', color='none',markeredgecolor='k', linestyle='none'),
        Line2D([0],[0], marker='s', color='none',markeredgecolor='k', linestyle='none')] + [
            Line2D([0],[0], linestyle='-', color = 3*[shades[r]])
            for r in range(len(rates))] + [Line2D([0],[0], color='k',linestyle='--')],
        ["Random Encodings", "Orthogonal Encodings"] + [
            "Success rate $\geq$ %.2f"%r for r in rates] + ["Heuristic"],
        loc='upper right')
    pt.xlabel("Line count")
    pt.ylabel("Network size")
    pt.tight_layout()
    pt.show()

def plot_match_trials_tokens(args_list, results, layer_shape_colors):
    sizes = {k: set() for k in [False, True]}
    token_counts = {k: set() for k in [False, True]}
    all_token_counts = []
    successes = {}
    failures = {}
    for a in range(len(args_list)):
        if results[a] is None: continue
        register_names, program = args_list[a][0], results[a][0]
        num_lines, layer_shape, orthogonal = args_list[a][3:6]
        leading_match_count, trial_step_count = results[a][1:3]
        lines, labels = preprocess(program, register_names)
        num_tokens = len(set([tok for line in lines for tok in line] + labels.keys()))
        success = (leading_match_count == trial_step_count)
        
        sizes[orthogonal].add(layer_shape[0])
        token_counts[orthogonal].add(num_tokens)
        all_token_counts.append(num_tokens)
        k = (layer_shape[0], num_tokens, orthogonal)
        if success: successes[k] = successes.get(k, 0) + 1
        else: failures[k] = failures.get(k, 0) + 1

    sizes = {k: np.sort(list(sizes[k])) for k in [False, True]}
    sizes[True] = sizes[True][[0, 2, -1]] # less cluttered figure
    token_counts = {k: np.sort(list(token_counts[k])) for k in [False, True]}
    x, y = {}, {}
    for orth in [False, True]:
        x[orth], y[orth] = {}, {}
        for size in sizes[orth]:
            x[orth][size], y[orth][size] = [], []
            for count in token_counts[orth]:
                k = (size, count, orth)
                success_count = float(successes.get(k,0))
                total_count = successes.get(k,0)+failures.get(k,0)
                if total_count == 0: continue
                x[orth][size].append(count)
                y[orth][size].append(success_count/total_count)

    xmx = max(token_counts[True].max(), token_counts[False].max())
    pt.figure(figsize=(7,5))
    pt.subplot(2,1,1)
    print("rand")
    h = []
    shades = np.linspace(0,0.5,len(sizes[False]))[::-1]
    for s, size in enumerate(sizes[False]):
        print(size, ["%d:%.2f"%tup for tup in zip(x[False][size], y[False][size])])
        h.append(pt.plot(x[False][size], y[False][size], '-o', color=3*[shades[s]])[0])
    pt.legend(h, ["N=%d"%s for s in sizes[False]])
    pt.ylabel("Success rate")
    pt.xlim([0, xmx])
    pt.title("Random Encodings")
    pt.subplot(2,1,2)
    print("orth")
    h = []
    shades = np.linspace(0,0.5,len(sizes[True]))[::-1]
    for s,size in enumerate(sizes[True]):
        print(size, ["%d:%.2f"%tup for tup in zip(x[True][size], y[True][size])])
        h.append(pt.plot(x[True][size], y[True][size], '-o', color=3*[shades[s]])[0])
    pt.legend(h, ["N=%d"%s for s in sizes[True]], loc='center')
    pt.ylabel("Success rate")
    pt.xlabel("Distinct token counts")
    pt.xlim([0, xmx])
    pt.title("Orthogonal Encodings")
    pt.tight_layout()
    pt.show()

    # To assess whether relationship between prog size and required nvm size linear, log, poly, etc:
    # for each line count, find smallest nvm size with perfect success rate.
    # then plot that line count vs required nvm size.
    h = []
    # rates = [.5, .8, .95, 1.]
    rates = [1.]
    shades = np.linspace(0, .5, len(rates))[::-1]
    pt.figure(figsize=(7,5))
    pt.subplot(2,1,1)
    for orth in [False, True]:
        mk = 's' if orth else 'o'
        for r,rate in enumerate(rates):
            x, y = [], []
            for count in token_counts[orth]:
                x.append(count)
                best_size = np.inf
                for size in sizes[orth]:
                    k = (size, count, orth)
                    success_count = float(successes.get(k,0))
                    total_count = successes.get(k,0)+failures.get(k,0)
                    print(k, success_count, total_count)
                    if total_count == 0: continue
                    success_rate = success_count/total_count
                    # if success_rate == 1.0 and size < best_size: best_size = size
                    if success_rate >= rate and size < best_size: best_size = size
                y.append(best_size)
            # h.append(pt.plot(x, y, '-', marker=mk, color=3*[shades[r]],markerfacecolor='none')[0])
            col = 0. if orth else .5
            h.append(pt.plot(x, y, '-', marker=mk, color=3*[col],markerfacecolor='none')[0])
    pt.legend(h, ["Random Encodings", "Orthogonal Encodings"])
    # pt.legend([
    #     Line2D([0],[0], marker='o', color='none',markeredgecolor='k', linestyle='none'),
    #     Line2D([0],[0], marker='s', color='none',markeredgecolor='k', linestyle='none')] + [
    #         Line2D([0],[0], linestyle='-', color = 3*[shades[r]])
    #         for r in range(len(rates))],
    #     ["Random Encodings", "Orthogonal Encodings"] + [
    #         "Success rate $\geq$ %.2f"%r for r in rates])
    pt.xlabel("Distinct token count")
    pt.ylabel("Smallest perfect network size")

    pt.subplot(2,1,2)
    distinct_token_counts = np.sort(list(set(all_token_counts)))
    count_frequencies = {tc: 0 for tc in distinct_token_counts}
    for tc in all_token_counts: count_frequencies[tc] += 1
    count_frequencies = [count_frequencies[tc] for tc in distinct_token_counts]
    pt.bar(distinct_token_counts, count_frequencies, color='k')
    pt.xlabel("Distinct token count")
    pt.ylabel("Frequency")
    pt.tight_layout()
    pt.show()


# def plot_match_trials(args_list, results, layer_shape_colors):

#     for a in range(len(args_list)):
#         register_names, program = args_list[a][0], results[a][0]
#         num_lines, layer_shape, orthogonal = args_list[a][3:6]
#         leading_match_count, trial_step_count = results[a][1:3]
#         lines, labels = preprocess(program, register_names)
#         num_tokens = len(set([tok for line in lines for tok in line] + labels.keys()))
#         num_tokens += np.random.rand()
        
#         success = (leading_match_count == trial_step_count)
#         if  orthogonal and  success:
#             # pt.scatter(num_tokens, layer_shape[0], marker='o',facecolors='none',edgecolors='k')
#             pt.plot(num_tokens, layer_shape[0], marker='o',markerfacecolor='none',markeredgecolor='k', linestyle='none')
#         if  orthogonal and ~success:
#             # pt.scatter(num_tokens, layer_shape[0], marker='o',facecolors='none',edgecolors='gray')
#             pt.plot(num_tokens, layer_shape[0], marker='d',markerfacecolor='none',markeredgecolor='gray', linestyle='none')
#         if ~orthogonal and  success:
#             # pt.scatter(num_tokens, layer_shape[0], marker='+',c='k')
#             pt.plot(num_tokens, layer_shape[0], marker='+',c='k', linestyle='none')
#         if ~orthogonal and ~success:
#             # pt.scatter(num_tokens, layer_shape[0], marker='+',c='gray')
#             pt.plot(num_tokens, layer_shape[0], marker='^',c='gray', linestyle='none')
#         # if  orthogonal and  success: kwa = dict(marker='o',c='k')
#         # if  orthogonal and ~success: kwa = dict(marker='o')
#         # if ~orthogonal and  success: kwa = dict(marker='+',c='k')
#         # if ~orthogonal and ~success: kwa = dict(marker='+',c='k')
#         # # if orthogonal and success: pt.scatter([0],[0],marker='o',c='k')
#         # if orthogonal and success:
#         #     print(num_tokens, layer_shape[0])
#         #     pt.scatter(num_tokens, layer_shape[0], **kwa)

#     pt.legend([
#         Line2D([0],[0], marker='o', color='none',markeredgecolor='k', linestyle='none'),
#         Line2D([0],[0], marker='+', color='k', linestyle='none'),
#         Line2D([0],[0], marker='s', color='k', linestyle='none'),
#         Line2D([0],[0], marker='s', color='gray', linestyle='none'),
#     ],["Orth","Rand","Success","Fail"])
#     pt.xlabel("# unique program tokens")
#     pt.ylabel("Layer size") 
#     pt.show()


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
    max_steps = 100
    verbose = 0
    program_load_sizes = {False: # not orth
        [ # (num_subroutines, num_lines) ... for each program to be stored
        ((1, 8), (1, 4)),
        ((1, 8), (1, 8)),
        ((2, 16), (2, 8)),
        ((2, 16), (2, 16)),
        ((2, 32), (2, 8)),
        ((2, 32), (2, 16)),
        ((2, 32), (2, 16), (1, 8)),
        ((3, 64), (2, 16), (2, 16)),
        ((3, 64), (2, 32), (2, 32)),
        ], True: [# orth
        ((1, 8), (1, 4)),
        ((1, 8), (1, 8)),
        ((2, 16), (2, 8)),
        ((2, 16), (2, 16)),
        ((2, 32), (2, 8)),
        ((2, 32), (2, 16)),
        ((2, 32), (2, 16), (1, 8)),
        ((3, 64), (2, 16), (2, 16)),
        ((4, 128), (3, 64), (2, 32)),
        ((5, 256), (4, 128), (3, 64)),
        ((8, 512), (5, 256), (5, 256)),
        ]}
    
    scaling = {
        False: np.array([.5]), # not orth
        True: np.array([]), #  orth
    }
        
    
    # layer_shapes = {False: [# not orth
    #     (16, 1),
    #     (32, 1),
    #     (64, 1),
    #     (128, 1),
    #     (256, 1),
    #     (512, 1),
    #     (1024, 1),
    #     ], True: [ # orth
    #     (16, 1),
    #     (32, 1),
    #     (64, 1),
    #     (256, 1),
    #     (512, 1),
    #     (1024, 1),
    #     ]}
    # layer_shape_colors = {k:
    #     {s: np.arange(0, len(layer_shapes[k]), dtype=float)[::-1][i]/len(layer_shapes[k])
    #         for i,s in enumerate(layer_shapes[k])}
    #     for k in [False, True]}
    # # layer_shape_colors = {
    # #     (16, 1): .8,
    # #     (32, 1): .6,
    # #     (64, 1): .4,
    # #     (128,1): .2,
    # #     (256,1): .0,
    # #     # (512,1): .0,
    # # }
    num_repetitions = 50
    
    program_loads = {}
    for orth in [False, True]:
        program_loads[orth] = []
        for program_load_size in program_load_sizes[orth]:
            program_load = []
            for (num_subroutines, num_lines) in program_load_size:
                program, tokens, initial_activity = generate_random_program(
                        register_names, num_tokens, num_subroutines, num_lines)
                program_load.append((program, tokens, initial_activity))
            program_loads[orth].append(program_load)

    # args:
    # register_names, program_load, layer_shape, shapes, orthogonal, max_steps, verbose=0
    args_list = []
    for orth in [False, True]:
        for program_load in program_loads[orth]:
            for scale in scaling[orth]:
                args_list.append((
                    register_names, program_load, layer_shape, shapes, orth))

    # args_list = [(register_names, program_load),it.product(program_loads[orth], layer_shapes[orth])

    # args_list = [(
    #     register_names, program_loads,
    #     layer_shape, orthogonal, max_steps, verbose)
    #     for orthogonal in [False, True]
    #         for ((num_subroutines, num_lines), layer_shape, _) in it.product(
    #             program_sizes[orthogonal],
    #             layer_shapes[orthogonal],
    #             range(num_repetitions))]

    plot_trial_complexities(register_names, program_loads[False] + program_loads[True])

    # print_random_program(register_names, num_tokens=num_tokens, num_subroutines=2, num_lines=12)

    # # results = run_match_trial_pool(args_list, num_procs=0)
    # # with open('tmp.pkl','w') as f: pk.dump((args_list, results), f)

    # # with open('tmp.pkl','r') as f: args_list, results = pk.load(f)
    # # with open('big_randprog.pkl','r') as f: args_list, results = pk.load(f)
    
    # # for a, res in enumerate(results):
    # #     if res is None:
    # #         print("crashed")
    # #         continue
    # #     (_, leading_match_count, trial_step_count, _, _) = res
    # #     print("|prog|=%d, N=%d, %s: %d of %d"%(
    # #         args_list[a][3], args_list[a][4][0],
    # #         "orth" if args_list[a][5] else "rand",
    # #         leading_match_count, trial_step_count))

    # # # plot_trial_complexities(args_list, results)
    # # plot_match_trials(args_list, results, layer_shape_colors)
    # # # plot_match_trials_tokens(args_list, results, layer_shape_colors)

    # # num_registers = 3
    # # register_names = ["r%d"%r for r in range(num_registers)]
    # # num_tokens = 1
    # # num_subroutines = 2
    # # num_lines = 10
    # # layer_shape = (32,32)
    # # orthogonal=True
    # # max_steps = 5

    # # program, leading_match_count, trial_step_count, nvm_trace, rvm_trace = match_trial(
    # #     register_names, num_tokens, num_subroutines, num_lines,
    # #     layer_shape, orthogonal, max_steps, verbose=0)

    # # for t in nvm_trace: print(t)
    # # print(program)
    # # for t in rvm_trace: print(t)
    # # print("matched leading %d of %d"%(leading_match_count, trial_step_count))
