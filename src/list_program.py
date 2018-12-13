import numpy as np
from nvm import make_scaled_nvm

list_programs = {"echo":"""

        ref rloc        # save starting memory location
loop1:  mov rval rinp   # stage current element
        mem rval        # store current element
        nxt             # advance to next memory location
        cmp rval nil    # check for nil terminator
        jie out         # if nil, leave loop 1
        mov rinp sep    # separator for IO protocol
        jmp loop1       # repeat

out:    drf rloc        # restore starting memory location
loop2:  rem rval        # retrieve current element
        mov rout rval   # output current element
        nxt             # advance to next memory location
        cmp rval nil    # check for nil terminator
        jie done        # if nil, leave loop 2
        mov rout sep    # separator for IO protocol
        jmp loop2       # repeat

done:   exit            # halt execution
    
"""}

if __name__ == "__main__":

    list_items = list("ABCDEF")

    nvm = make_scaled_nvm(
        register_names = ["rinp","rout","rloc","rval"],
        programs = list_programs,
        orthogonal=True,
        extra_tokens=list_items + ["nil", "sep"])
    nvm.assemble(list_programs)
    
    list_length = 10
    list_index = 0
    list_input = list(np.random.choice(list_items, list_length)) + ["nil"]
    list_output = []
    rout_was_sep = True
    
    nvm.load("echo", {
        "rinp": list_input[list_index],
        "rout": "sep",
        "rloc": "adr"
    })

    show_layers = [
        ["go", "gh","ip"] + ["op"+x for x in "c12"] +\
        ["mf","mb"] + ["sf","sb"] + ["co","ci"] +\
        nvm.net.devices.keys(),
    ]
    show_tokens = True
    show_corrosion = False
    show_gates = False
    
    for t in range(500):
    
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
        nvm.step(verbose=1)
        if nvm.at_exit(): break

    print("%d steps:" % t)

    print("list input:")
    print(list_input)

    print("list output:")
    print(list_output)

