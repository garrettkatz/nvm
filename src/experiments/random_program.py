"""
Used to generate Figures 12-14 in the 2019 NVM paper.
"""
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
from matplotlib.lines import Line2D
import multiprocessing as mp
import itertools as it
from nvm import make_scaled_nvm
from preprocessing import *
from refvm import RefVM

np.set_printoptions(formatter={'float': (lambda x: '%.3f'%x)}, linewidth=200)

hr_opcodes = np.array([
    "mov","jmp",
    "cmp","jie",
    "sub",
    "mem","nxt","prv","ref",
    # "rem","drf",
    ], dtype=object)

def generate_random_program(name, register_names, num_tokens, num_subroutines, num_lines):

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
    labels = {"%s.lab%d"%(name,l): l for l in subroutine_partition[1:-1]}

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
            label = "%s.lab%d"%(name,label_line)
            lines[l] += " %s"%label
            labels[label] = label_line # save new label
        if lines[l] == "sub":
            label = "%s.lab%d"%(name,np.random.choice(subroutine_partition[1:-1]))
            lines[l] += " %s"%label
        if lines[l] in ["mem","rem","ref","drf"]:
            lines[l] += " %s"%np.random.choice(register_names)

    # Prepend labels to program lines
    for l in range(len(lines)):
        lines[l] = " "*16 + lines[l]
    for label, l in labels.items():
        lines[l] = label + ":" + lines[l][len(label)+1:]
    
    program = "\n".join(lines)
    return program

def print_random_program(register_names, num_tokens, num_subroutines, num_lines):
    program = generate_random_program("rand",register_names, num_tokens, num_subroutines, num_lines)
    lines, labels, tokens = preprocess({"rand": program}, register_names)
    lines, labels = lines["rand"], labels["rand"]
    distinct_tokens = list(tokens) + labels.keys()
    print(program)
    print("")
    print(list(distinct_tokens))
    print("%d distinct tokens"%len(distinct_tokens))

def run_program(vm, program, name, initial_activity, max_steps, verbose=0):

    # Load and run
    # vm.assemble(program, name, verbose=0, other_tokens=tokens)
    vm.load(name, initial_activity)

    trace = []
    for t in range(max_steps):
        if vm.at_exit(): break
        if verbose > 0: print("step %d: %s"%(t, vm.state_string()))
        vm.step(verbose=0, max_ticks=20)
        trace.append(vm.decode_state(layer_names=[
            "opc","op1","op2","co"] + vm.register_names))
    
    return trace

def match_trial(
    register_names,
    programs,
    initial_activities,
    extra_tokens,
    scale_factor,
    orthogonal,
    max_steps,
    verbose=0):

    # Make vms and assemble
    nvm = make_scaled_nvm(
        register_names, programs, orthogonal=orthogonal,
        scale_factor=scale_factor, extra_tokens=extra_tokens)
    rvm = RefVM(register_names)

    nvm.assemble(programs, other_tokens=extra_tokens, verbose=verbose)
    rvm.assemble(programs, other_tokens=extra_tokens, verbose=verbose)

    # Load and run programs
    leading_match_counts, trial_step_counts, nvm_traces, rvm_traces = [], [], [], []
    for name, program in programs.items():
    
        nvm_trace = run_program(nvm,
            program, name, initial_activities[name], max_steps=max_steps, verbose=0)
        rvm_trace = run_program(rvm,
            program, name, initial_activities[name], max_steps=max_steps, verbose=0)
    
        for t in range(len(rvm_trace)):
            if t >= len(nvm_trace): break
            if nvm_trace[t] != rvm_trace[t]: break
        
        leading_match_count = t+1
        trial_step_count = len(rvm_trace)
        leading_match_counts.append(leading_match_count)
        trial_step_counts.append(trial_step_count)
        nvm_traces.append(nvm_trace)
        rvm_traces.append(rvm_trace)
    
    layer_size = nvm.net.layers[register_names[0]].size
    ip_size = nvm.net.layers['ip'].size
    
    return leading_match_counts, trial_step_counts, nvm_traces, rvm_traces, layer_size, ip_size

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

        programs = {"rand%d"%p: program
            for p, (program, _, _) in enumerate(program_load)}
        lines, labels, tokens = preprocess(programs, register_names)
        num_lines = sum([len(lines[name]) for name in lines])
        all_labels = set()
        for name in labels:
            all_labels = all_labels | set(labels[name].keys())
        num_tokens = len(tokens | all_labels | set(register_names))
        
        line_counts.append(num_lines)
        token_counts.append(num_tokens)

    lc = np.array(line_counts)
    tc = np.array(token_counts)
    
    print(np.array([lc, tc]).T)

    pt.figure(figsize=(6.5,2))
    pt.scatter(lc, tc, c='k', marker='+')
    # pt.scatter(lc + 4*np.random.random_sample(*lc.shape), tc, c='k', marker='+')
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

def plot_asymptotic_scaling(args_list, results, use_ip=False):
    # args: register_names, programs, initial_activities, extra_tokens, scale_factor, orthogonal, max_steps
    # res: leading_match_counts, trial_step_counts, nvm_traces, rvm_traces, layer_size, ip_size
    
    # plot pattern counts against the lowest scale factors with perfect (or near) success rate
    # need to get all success rates for all scale factors at each pattern count, then aggregate
    success_flags = {True:{}, False:{}}
    for args, res in zip(args_list, results):
        if res is None: continue
        
        register_names, programs, _, extra_tokens, scale_factor, orthogonal, _ = args
        leading_match_counts, trial_step_counts, _, _, layer_size, ip_size = res
        
        if use_ip:
            prog_size, _, _ = measure_programs(
                programs, register_names, extra_tokens=extra_tokens)
            scale_factor = float(ip_size)/prog_size
        else:
            _, prog_size, _ = measure_programs(
                programs, register_names, extra_tokens=extra_tokens)
            scale_factor = float(layer_size)/prog_size
        # num_lines, num_tokens, _ = measure_programs(
        #     programs, register_names, extra_tokens=extra_tokens)
        # scale_factor = float(layer_size)/num_tokens
        # prog_size = (num_lines, num_tokens)

        success = all([(lc == tc) for lc, tc in zip(
            leading_match_counts, trial_step_counts)])

        if prog_size not in success_flags[orthogonal]:
            success_flags[orthogonal][prog_size] = {}
        if scale_factor not in success_flags[orthogonal][prog_size]:
            success_flags[orthogonal][prog_size][scale_factor] = []

        success_flags[orthogonal][prog_size][scale_factor].append(success)
    
    success_rates = {}
    for orthogonal in success_flags:
        success_rates[orthogonal] = {}
        for prog_size in success_flags[orthogonal]:
            success_rates[orthogonal][prog_size] = {}
            for scale_factor in success_flags[orthogonal][prog_size]:
                flags = success_flags[orthogonal][prog_size][scale_factor]
                rate = float(sum(flags))/len(flags)
                success_rates[orthogonal][prog_size][scale_factor] = rate
                
    # rate_bounds = [0.9, 1.]
    rate_bounds = [1., .9]
    lowest_scales = {}
    for orthogonal in success_rates:
        lowest_scales[orthogonal] = {}
        for rate_bound in rate_bounds:
            lowest_scales[orthogonal][rate_bound] = {}
            for prog_size in success_rates[orthogonal]:
                low_scale = np.inf
                for scale, rate in success_rates[orthogonal][prog_size].items():
                    if rate >= rate_bound and scale < low_scale:
                        low_scale = scale
                lowest_scales[orthogonal][rate_bound][prog_size] = low_scale
    
    pt.figure(figsize=(9,4))
    pt.subplot(1,2,1)
    h, leg = [], []
    for orthogonal in lowest_scales:
        marker = 'd' if orthogonal else 'o'
        for rate_bound in rate_bounds:
            x, y = zip(*lowest_scales[orthogonal][rate_bound].items())
            c = .75*(rate_bound - min(rate_bounds))/(max(rate_bounds)-min(rate_bounds))
            # c = 0
            h.append( pt.plot(x, y, marker, color=3*[c], mfc='none')[0] )
            leg.append( ("Orthogonal" if orthogonal else "Bernoulli") + "$\geq %.3f$"%rate_bound )
    pt.legend(h, leg)
    pt.plot([0, 256],[1, 1],'--k')
    pt.plot([0, 256],[20, 20],'--k')
    if use_ip:
        pt.xlabel("Number of lines")
    else:
        pt.xlabel("Number of distinct symbols")
    pt.ylabel("Scale factor")

    # plot pattern counts against the lowest layer sizes with perfect (or near) success rate
    pt.subplot(1,2,2)
    h, leg = [], []
    for orthogonal in lowest_scales:
        marker = 'd' if orthogonal else 'o'
        for rate_bound in rate_bounds:
            x, y = zip(*lowest_scales[orthogonal][rate_bound].items())
            y = np.array(y) * np.array(x)
            c = .75*(rate_bound - min(rate_bounds))/(max(rate_bounds)-min(rate_bounds))
            # c = 0
            h.append( pt.plot(x, y, marker, color=3*[c], mfc='none')[0] )
            leg.append( ("Orthogonal" if orthogonal else "Bernoulli") + "$\geq %.3f$"%rate_bound )
    pt.legend(h, leg)
    if use_ip:
        pt.xlabel("Number of lines")
        pt.ylabel("ip size")
    else:
        pt.xlabel("Number of distinct symbols")
        pt.ylabel("Register size")

    pt.tight_layout()
    pt.show()


def plot_match_trials_tokens(args_list, results):

    # args: register_names, programs, initial_activities, extra_tokens, scale_factor, orthogonal, max_steps
    # res: leading_match_counts, trial_step_counts, nvm_traces, rvm_traces, layer_size, ip_size
    
    # data[orth][scale_factor][num_patterns] = 0/1 list of successful trials with that orth, scale, count
    data = {orth:{} for orth in [True, False]}

    for args, res in zip(args_list, results):
        if res is None: continue
        
        register_names, programs, _, extra_tokens, scale_factor, orthogonal, _ = args
        leading_match_counts, trial_step_counts, _, _, layer_size, ip_size = res
        
        _, num_patterns, _ = measure_programs(
            programs, register_names, extra_tokens=extra_tokens)
        scale_factor = float(layer_size)/num_patterns

        success = all([(lc == tc) for lc, tc in zip(
            leading_match_counts, trial_step_counts)])

        if scale_factor not in data[orthogonal]:
            data[orthogonal][scale_factor] = {}
        if num_patterns not in data[orthogonal][scale_factor]:
            data[orthogonal][scale_factor][num_patterns] = []

        data[orthogonal][scale_factor][num_patterns].append(success)

    pt.figure(figsize=(7,5))
    for o, orthogonal in enumerate([True, False]):
        pt.subplot(2,1,o+1)
        pt.title('Orth' if orthogonal else 'Rand')
        max_scale = np.max(data[orthogonal].keys())
        for scale_factor in data[orthogonal]:
            x, y = [], []
            for num_patterns, successes in data[orthogonal][scale_factor].items():
                x.append(num_patterns)
                y.append(float(sum(successes))/len(successes))
            print(orthogonal, scale_factor)
            print(np.array([x,y]))
            pt.plot(x, y, '-+', color=3*[scale_factor/(1.5*max_scale)])

    pt.show()
    
    # scatter plot: scale factor vs success rate
    pt.figure(figsize=(7,5))
    for o, orthogonal in enumerate([True, False]):
        pt.subplot(2,1,o+1)
        pt.title('Orth' if orthogonal else 'Rand')
        max_scale = np.max(data[orthogonal].keys())
        x, y = [], []
        for scale_factor in data[orthogonal]:
            for num_patterns, successes in data[orthogonal][scale_factor].items():
                success_rate = float(sum(successes))/len(successes)
                x.append(scale_factor)
                y.append(success_rate)
        pt.plot(x, y, 'o')
    pt.show()
    
    # To assess relationship between prog size and required nvm size:
    # for each token count, find smallest nvm size with perfect success rate.
    # then plot that pattern count vs required nvm size.
    # data[orth][line_count][layer_size] = 0/1 list of successful trials with that orth, count, size
    # data = {orth:{} for orth in [True, False]}
    # for args, res in zip(args_list, results):
    #     if res is None: continue
        
    #     register_names, programs, _, extra_tokens, scale_factor, orthogonal, _ = args
    #     leading_match_counts, trial_step_counts, _, _, layer_size, ip_size = res
        
    #     _, num_patterns, _ = measure_programs(
    #         programs, register_names, extra_tokens=extra_tokens)
    #     scale_factor = float(layer_size)/num_patterns

    #     success = all([(lc == tc) for lc, tc in zip(
    #         leading_match_counts, trial_step_counts)])

    #     if scale_factor not in data[orthogonal]:
    #         data[orthogonal][scale_factor] = {}
    #     if num_patterns not in data[orthogonal][scale_factor]:
    #         data[orthogonal][scale_factor][num_patterns] = []

    #     data[orthogonal][scale_factor][num_patterns].append(success)                 
    
    # rates = [.5, .8, .95, 1.]
    # shades = [.7, .6, .1, 0] #np.linspace(0, .5, len(rates))[::-1]
    # h = []
    # pt.figure(figsize=(7,3.5))
    # for orth in [False, True]:
    #     mk = 's' if orth else 'o'
    #     for r,rate in enumerate(rates):
    #         x, y = [], []
    #         for scale_factor in data[orth]:
    #             for num_patterns in data[orth][scale_factor]:
    #                 x.append(num_patterns)
    #                 best_size = np.inf
    #                 for size in sizes[orth]:
    #                     k = (size, count, orth)
    #                     success_count = float(successes.get(k,0))
    #                     total_count = successes.get(k,0)+failures.get(k,0)
    #                     print(k, success_count, total_count)
    #                     if total_count == 0: continue
    #                     success_rate = success_count/total_count
    #                     # if success_rate == 1.0 and size < best_size: best_size = size
    #                     if success_rate >= rate and size < best_size: best_size = size
    #                 y.append(best_size)
    #         h.append(pt.plot(x, y, '-', marker=mk, color=3*[shades[r]],markerfacecolor='none')[0])
    # # fit logarithm: C log2(256) = 256 -> C = 256/log2(256) = 256/8 = 32
    # # fit sqrt: C sqrt(256) = 256 -> C = 256/sqrt(256) = 16
    # pt.plot(np.arange(1,1000), 32*np.log2(np.arange(1,1000)), '--k')
    # # pt.plot(np.arange(1,1000), 16*np.sqrt(np.arange(1,1000)), '--k')
    # pt.legend([
    #     Line2D([0],[0], marker='o', color='none',markeredgecolor='k', linestyle='none'),
    #     Line2D([0],[0], marker='s', color='none',markeredgecolor='k', linestyle='none')] + [
    #         Line2D([0],[0], linestyle='-', color = 3*[shades[r]])
    #         for r in range(len(rates))] + [Line2D([0],[0], color='k',linestyle='--')],
    #     ["Random Encodings", "Orthogonal Encodings"] + [
    #         "Success rate $\geq$ %.2f"%r for r in rates] + ["Heuristic"],
    #     loc='upper right')
    # pt.xlabel("Line count")
    # pt.ylabel("Network size")
    # pt.tight_layout()
    # pt.show()
    
    # xmx = max(token_counts[True].max(), token_counts[False].max())
    # pt.figure(figsize=(7,5))
    # pt.subplot(2,1,1)
    # print("rand")
    # h = []
    # shades = np.linspace(0,0.5,len(sizes[False]))[::-1]
    # for s, size in enumerate(sizes[False]):
    #     print(size, ["%d:%.2f"%tup for tup in zip(x[False][size], y[False][size])])
    #     h.append(pt.plot(x[False][size], y[False][size], '-o', color=3*[shades[s]])[0])
    # pt.legend(h, ["N=%d"%s for s in sizes[False]])
    # pt.ylabel("Success rate")
    # pt.xlim([0, xmx])
    # pt.title("Random Encodings")
    # pt.subplot(2,1,2)
    # print("orth")
    # h = []
    # shades = np.linspace(0,0.5,len(sizes[True]))[::-1]
    # for s,size in enumerate(sizes[True]):
    #     print(size, ["%d:%.2f"%tup for tup in zip(x[True][size], y[True][size])])
    #     h.append(pt.plot(x[True][size], y[True][size], '-o', color=3*[shades[s]])[0])
    # pt.legend(h, ["N=%d"%s for s in sizes[True]], loc='center')
    # pt.ylabel("Success rate")
    # pt.xlabel("Distinct token counts")
    # pt.xlim([0, xmx])
    # pt.title("Orthogonal Encodings")
    # pt.tight_layout()
    # pt.show()

    # # To assess whether relationship between prog size and required nvm size linear, log, poly, etc:
    # # for each line count, find smallest nvm size with perfect success rate.
    # # then plot that line count vs required nvm size.
    # h = []
    # # rates = [.5, .8, .95, 1.]
    # rates = [1.]
    # shades = np.linspace(0, .5, len(rates))[::-1]
    # pt.figure(figsize=(7,5))
    # pt.subplot(2,1,1)
    # for orth in [False, True]:
    #     mk = 's' if orth else 'o'
    #     for r,rate in enumerate(rates):
    #         x, y = [], []
    #         for count in token_counts[orth]:
    #             x.append(count)
    #             best_size = np.inf
    #             for size in sizes[orth]:
    #                 k = (size, count, orth)
    #                 success_count = float(successes.get(k,0))
    #                 total_count = successes.get(k,0)+failures.get(k,0)
    #                 print(k, success_count, total_count)
    #                 if total_count == 0: continue
    #                 success_rate = success_count/total_count
    #                 # if success_rate == 1.0 and size < best_size: best_size = size
    #                 if success_rate >= rate and size < best_size: best_size = size
    #             y.append(best_size)
    #         # h.append(pt.plot(x, y, '-', marker=mk, color=3*[shades[r]],markerfacecolor='none')[0])
    #         col = 0. if orth else .5
    #         h.append(pt.plot(x, y, '-', marker=mk, color=3*[col],markerfacecolor='none')[0])
    # pt.legend(h, ["Random Encodings", "Orthogonal Encodings"])
    # # pt.legend([
    # #     Line2D([0],[0], marker='o', color='none',markeredgecolor='k', linestyle='none'),
    # #     Line2D([0],[0], marker='s', color='none',markeredgecolor='k', linestyle='none')] + [
    # #         Line2D([0],[0], linestyle='-', color = 3*[shades[r]])
    # #         for r in range(len(rates))],
    # #     ["Random Encodings", "Orthogonal Encodings"] + [
    # #         "Success rate $\geq$ %.2f"%r for r in rates])
    # pt.xlabel("Distinct token count")
    # pt.ylabel("Smallest perfect network size")

    # pt.subplot(2,1,2)
    # distinct_token_counts = np.sort(list(set(all_token_counts)))
    # count_frequencies = {tc: 0 for tc in distinct_token_counts}
    # for tc in all_token_counts: count_frequencies[tc] += 1
    # count_frequencies = [count_frequencies[tc] for tc in distinct_token_counts]
    # pt.bar(distinct_token_counts, count_frequencies, color='k')
    # pt.xlabel("Distinct token count")
    # pt.ylabel("Frequency")
    # pt.tight_layout()
    # pt.show()


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
    extra_tokens = ["t%d"%r for t in range(num_tokens)]

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
        ((4, 128), (3, 32), (2, 16)),
        ((4, 128), (3, 32), (2, 32)),
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
    
    num_repetitions = 30
    
    # print_random_program(register_names, num_tokens=num_tokens, num_subroutines=2, num_lines=12)
    # plot_trial_complexities(register_names, program_loads[False] + program_loads[True])

    scaling = {
        False: np.array([.75, 1., 1.25, 1.5]), # not orth
        True: np.array([.5, 1., 2.]), #  orth
        # False: np.array([1, 1.5]), # not orth
        # True: np.array([.5, 1]), #  orth
    }
        
    # args:
    # register_names, programs, initial_activities, extra_tokens, scale_factor, orthogonal, max_steps
    args_list = []
    for orthogonal in [False, True]:
        for program_load_size in program_load_sizes[orthogonal]:
            for rep in range(num_repetitions):

                programs, initial_activities = {}, {}
                for p,(num_subroutines, num_lines) in enumerate(program_load_size):
                    name = "rand%d"%p
                    programs[name] = generate_random_program(
                        name, register_names, num_tokens, num_subroutines, num_lines)
                    initial_activities[name] = {
                        r: np.random.choice(extra_tokens) for r in register_names}
            
                for scale_factor in scaling[orthogonal]:
                    args_list.append((
                        register_names, programs, initial_activities,
                        extra_tokens, scale_factor, orthogonal, max_steps))

    num_procs = 6
    # results = run_match_trial_pool(args_list, num_procs=num_procs)
    # with open('tmp.pkl','w') as f: pk.dump((args_list, results), f)

    # with open('tmp.pkl','r') as f: args_list, results = pk.load(f)
    # with open('big_randprog.pkl','r') as f: args_list, results = pk.load(f)
    with open('big_new.pkl','r') as f: args_list, results = pk.load(f)
    
    # args: register_names, programs, initial_activities, extra_tokens, scale_factor, orthogonal, max_steps
    # res: leading_match_counts, trial_step_counts, nvm_traces, rvm_traces, layer_size, ip_size
    for args, res in zip(args_list, results):

        register_names, programs, initial_activities, extra_tokens, scale_factor, orthogonal, _ = args
        num_lines, num_patterns, _ = measure_programs(programs, register_names, extra_tokens)

        if res is None:
            print(
                "%d lines, %d tokens, %f scale, %s: crashed"%(
                    num_lines, num_patterns, scale_factor, "orth" if orthogonal else "rand"))
            continue

        leading_match_counts, trial_step_counts, nvm_traces, rvm_traces, layer_size, ip_size = res
        # if not orthogonal:
        if True:
            print(
                "%d lines, %d tokens, %f scale, %d layer, %d ip, %s: "%(
                    num_lines, num_patterns, scale_factor,
                    layer_size, ip_size, "orth" if orthogonal else "rand") + \
                ", ".join(["%d of %d"%count for count in zip(leading_match_counts, trial_step_counts)]))
            for p in range(len(leading_match_counts)):
                t = leading_match_counts[p]
                if t < trial_step_counts[p]:
                    print(t)
                    if t > 1:
                        print('nvm-1', 'rvm-1')
                        print(nvm_traces[p][t-2])
                        print(rvm_traces[p][t-2])
                    print('nvm', 'rvm')
                    if t < len(nvm_traces[p]):                    
                        print(nvm_traces[p][t-1])
                    else: print('nvm stopped')
                    print(rvm_traces[p][t-1])
        
        fail = False
        for p,count in enumerate(zip(leading_match_counts, trial_step_counts)):
            if orthogonal and count[0] != count[1]: fail = True

        if fail:
            for p in range(len(programs)):
                pass
                # # print(p)
                # print(programs["rand%d"%p])

    # plot_trial_complexities(register_names, results)
    # # plot_match_trials(args_list, results, layer_shape_colors)
    # plot_match_trials_tokens(args_list, results)
    plot_asymptotic_scaling(args_list, results, use_ip=True)
    
    # # # num_registers = 3
    # # # register_names = ["r%d"%r for r in range(num_registers)]
    # # # num_tokens = 1
    # # # num_subroutines = 2
    # # # num_lines = 10
    # # # layer_shape = (32,32)
    # # # orthogonal=True
    # # # max_steps = 5

    # # # program, leading_match_count, trial_step_count, nvm_trace, rvm_trace = match_trial(
    # # #     register_names, num_tokens, num_subroutines, num_lines,
    # # #     layer_shape, orthogonal, max_steps, verbose=0)

    # # # for t in nvm_trace: print(t)
    # # # print(program)
    # # # for t in rvm_trace: print(t)
    # # # print("matched leading %d of %d"%(leading_match_count, trial_step_count))
