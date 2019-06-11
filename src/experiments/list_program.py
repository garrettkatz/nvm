"""
Used to generate Figure 11 in the 2019 NVM paper
"""
import numpy as np
import pickle as pk
import matplotlib.pyplot as pt
from nvm import make_scaled_nvm

list_programs = {"echo":"""

        ref rloc        # save starting memory location
loop1:  mov rval rinp   # stage current element
        mem rval        # store current element
        nxt             # advance to next memory location
        cmp rval end    # check for list terminator
        jie out         # if list ended, leave loop 1
        mov rinp sep    # separator for IO protocol
        jmp loop1       # repeat

out:    drf rloc        # restore starting memory location
loop2:  rem rval        # retrieve current element
        mov rout rval   # output current element
        nxt             # advance to next memory location
        cmp rval end    # check for list terminator
        jie done        # if list ended, leave loop 2
        mov rout sep    # separator for IO protocol
        jmp loop2       # repeat

done:   exit            # halt execution
    
"""}

def run_trial(num_items, list_length, orth, scale_factor, verbose=False):

    list_items = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:num_items]

    nvm = make_scaled_nvm(
        register_names = ["rinp","rout","rloc","rval"],
        programs = list_programs,
        orthogonal=orth,
        scale_factor=scale_factor,
        extra_tokens=list_items + ["end", "sep"],
        num_addresses=list_length+1)
    nvm.assemble(list_programs)
    
    list_input = list(np.random.choice(list_items, list_length)) + ["end"]
    list_output = []
    list_index = 0
    rout_was_sep = True
    
    nvm.load("echo", {
        "rinp": list_input[list_index],
        "rout": "sep",
        "rloc": "adr"
    })

    for t in range(700): # length 50 requires >= 698 steps
    
        # input
        if nvm.decode_layer("rinp") == "sep":
            list_index += 1
            if list_index == len(list_input): break
            nvm.encode_symbol("rinp", list_input[list_index])

        # output
        rout = nvm.decode_layer("rout")
        if rout_was_sep and rout != "sep":
            list_output.append(rout)
            rout_was_sep = False
        if rout == "sep":
            rout_was_sep = True

        # step
        success = nvm.step(verbose=0)
        if not success: break
        if nvm.at_exit(): break

    if verbose:
        print("%d steps:" % t)
    
        print("list input:")
        print(list_input)
    
        print("list output:")
        print(list_output)
    
        print("equal:")
        print(list_input == list_output)
    
    matches = 0
    for i in range(min(len(list_input), len(list_output))):
        if list_input[i] == list_output[i]: matches += 1
    
    return t, matches

def plot_results(errs):
    
    # plot pattern counts against the lowest scale factors with perfect (or near) success rate
    # need to get all success rates for all scale factors at each pattern count, then aggregate
    
    pt.figure(figsize=(9,3))
    for orth in errs:

        data = {}
        for scale_factor in errs[orth]:
            data[scale_factor] = []

            for num_items in errs[orth][scale_factor]:
                for list_length in errs[orth][scale_factor][num_items]:
                    if list_length == 50: continue
                    reps = len(errs[orth][scale_factor][num_items])
                    successes = 0
                    match_rates = []
                    for r in range(reps):                    
                        _, t, matches, failed = errs[orth][scale_factor][num_items][list_length][r]
                        if not failed: successes += 1
                        match_rates.append(float(matches)/(list_length+1))
                    success_rate = float(successes)/reps
                    # data[scale_factor].append((list_length, success_rate))
                    data[scale_factor].append((list_length, np.mean(match_rates)))

        o = int(orth)
        pt.subplot(1,2,1+o)

        leg = []
        for scale_factor in errs[orth]:
            marker = 'o+d^'[errs[orth].keys().index(scale_factor)]
            c = 0 #1. - scale_factor/max(errs[orth].keys())
            x,y = zip(*data[scale_factor])
            idx = np.argsort(x)
            print(idx)
            x, y = np.array(x)[idx], np.array(y)[idx]
            pt.plot(x,y, marker=marker, color=3*[c], mfc='none', linestyle='-')
            leg.append("Scale = %.2f"%scale_factor)
        pt.legend(leg)
        pt.xlabel("List length")
        if o == 0:
            # pt.ylabel("Success rate")
            pt.ylabel("Average match rate")
        pt.ylim([-.1,1.1])
        pt.title("%s trials"%["Bernoulli","Orthogonal"][o])

    pt.tight_layout()
    pt.show()


if __name__ == "__main__":

    scaling = {
        False: np.array([.75, 1., 1.25, 1.5]), # not orth
        True: np.array([.75, 1., 1.25, 1.5]), #  orth
        # False: np.array([.5, 1.5]), # not orth
        # True: np.array([.5, 1]), #  orth
        # False: np.array([1.5]), # not orth
        # True: np.array([1]), #  orth
    }

    reps = 30
    
    if False: # Run trials
        errs = {}
        for orth in [False, True]:
            errs[orth] = {}
            for scale_factor in scaling[orth]:
                errs[orth][scale_factor] = {}
    
                # for num_items in range(1,27):
                for num_items in [26]:
                    errs[orth][scale_factor][num_items] = {}
    
                    for list_length in range(10,41,5):
                        errs[orth][scale_factor][num_items][list_length] = []
                    
                        args = (num_items, list_length, orth, scale_factor)
                        
                        for r in range(reps):
    
                            t, matches = run_trial(*args, verbose=False)
                            errs[orth][scale_factor][num_items][list_length].append(
                                (args, t, matches, matches != list_length+1))
                            
                            print("orth=%s, scale=%f, %d items, rep %d, %d steps, length %d =? %d matches"%(
                                orth, scale_factor, num_items, r, t, list_length+1, matches))
                            if matches != list_length+1: print("ERR!!!")
            
                            with open('lp.pkl','w') as f: pk.dump(errs, f)
    # analyze results
    if True:

        with open('lp.pkl','r') as f: errs = pk.load(f)

        for orth in errs:
            for scale_factor in errs[orth]:
                for num_items in errs[orth][scale_factor]:
                    for list_length in errs[orth][scale_factor][num_items]:
                        for r in range(reps):
    
                            _, t, matches, failed = errs[orth][scale_factor][num_items][list_length][r]
                            
                            print("orth=%s, scale=%f, %d items, rep %d, %d steps, length %d =? %d matches"%(
                                orth, scale_factor, num_items, r, t, list_length+1, matches) + \
                                (" ERR!!!" if failed else ""))

        plot_results(errs)
