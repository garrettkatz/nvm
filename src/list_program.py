import numpy as np
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

def run_trial(num_items, list_length, orth, verbose=False):

    list_items = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:num_items]

    nvm = make_scaled_nvm(
        register_names = ["rinp","rout","rloc","rval"],
        programs = list_programs,
        orthogonal=orth,
        extra_tokens=list_items + ["end", "sep"])
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

    for t in range(1000000):
    
        # input
        if nvm.decode_layer("rinp") == "sep":
            list_index += 1
            nvm.encode_symbol("rinp", list_input[list_index])

        # output
        rout = nvm.decode_layer("rout")
        if rout_was_sep and rout != "sep":
            list_output.append(rout)
            rout_was_sep = False
        if rout == "sep":
            rout_was_sep = True

        # step
        nvm.step(verbose=0)
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


if __name__ == "__main__":


    errs = {}
    for orth in [False, True]:
        errs[orth] = {}
        for num_items in range(26,27):
        
            for list_length in range(2,10):
            
                t, matches = run_trial(num_items, list_length, orth, verbose=False)
                errs[orth][list_length] = (matches != list_length+1)
                
                if matches != list_length+1: print("ERR!!!")
    
                print("orth=%s, %d items, length %d, %d steps, %d matches"%(
                    orth, num_items, list_length, t, matches))
