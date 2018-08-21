from nvm import make_scaled_nvm
from refvm import RefVM
import numpy as np
from random_program import run_program

programs = {"rand0":"""
                cmp r0 r2
                nxt
rand0.lab2:     jie rand0.lab5
rand0.lab3:     jie rand0.lab3
                jie rand0.lab2
rand0.lab5:     sub rand0.lab8
                ref r0
                exit
rand0.lab8:     cmp r2 r0
                mov r0 tok1
                sub rand0.lab15
rand0.lab11:    ref r0
                jie rand0.lab11
                ref r0
                ret
rand0.lab15:    ret
""","rand1":"""
                sub rand1.lab6
                mov r2 r0
                exit
rand1.lab3:     cmp r0 r0
rand1.lab4:     jie rand1.lab4
                ret
rand1.lab6:     nxt
                ret
"""}

num_registers = 3
register_names = ["r%d"%r for r in range(num_registers)]
tokens = ['tok'+str(t) for t in range(2)]

for rep in range(100):

    nvm = make_scaled_nvm(
        register_names, programs, orthogonal=True, scale_factor=20.0, extra_tokens=tokens)
    rvm = RefVM(register_names)
    
    nvm.assemble(programs, other_tokens=tokens)
    rvm.assemble(programs, other_tokens=tokens)
    
    for p in range(2):
        initial_activity = {r: np.random.choice(tokens) for r in register_names}
        
        nvm_trace = run_program(nvm,
            programs["rand%d"%p], "rand%d"%p, initial_activity, max_steps=100, verbose=0)
        rvm_trace = run_program(rvm,
            programs["rand%d"%p], "rand%d"%p, initial_activity, max_steps=100, verbose=0)
        
        fail = False
        for t in range(len(rvm_trace)):
            if t >= len(nvm_trace): break
            if nvm_trace[t] != rvm_trace[t]:
                fail = True
                for t2 in range(t+1):
                    print(rvm_trace[t2])
                    print(nvm_trace[t2])
                print(t, len(rvm_trace))
                break
        
        if fail: break

    if fail: break
