from nvm import make_scaled_nvm
from refvm import RefVM
import numpy as np
from random_program import run_program

programs = {"rand0":"""
                jie rand0.lab10
                mov r0 r2
                mov r1 r1
                ref r2
                sub rand0.lab21
                nxt
                jie rand0.lab14
                jmp rand0.lab13
                sub rand0.lab21
                mem r1
rand0.lab10:    sub rand0.lab19
                cmp r0 r2
                prv
rand0.lab13:    cmp r1 r0
rand0.lab14:    prv
                nxt
                prv
                nxt
                exit
rand0.lab19:    sub rand0.lab21
                ret
rand0.lab21:    cmp r2 r0
rand0.lab22:    ref r2
                cmp r1 tok0
                jie rand0.lab22
                cmp r2 r2
                nxt
                cmp r1 tok0
                mov r2 r1
                sub rand0.lab21
                ref r0
                ret
""", "rand1":"""
                mem r2
rand1.lab1:     mem r1
                jie rand1.lab1
                nxt
                mov r1 r0
                jmp rand1.lab1
                mem r2
                sub rand1.lab9
                exit
rand1.lab9:     sub rand1.lab9
                jmp rand1.lab11
rand1.lab11:    ret
rand1.lab12:    ref r0
                prv
                sub rand1.lab9
                ret
"""}

num_registers = 3
register_names = ["r%d"%r for r in range(num_registers)]
tokens = ['tok'+str(t) for t in range(2)]

for rep in range(100):

    nvm = make_scaled_nvm(
        register_names, programs, orthogonal=True, scale_factor=1.0, tokens=tokens)
    rvm = RefVM(register_names)
    
    nvm.assemble(programs)
    rvm.assemble(programs)
    
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
                print(t, len(rvm_trace))
                print(rvm_trace[t])
                print(nvm_trace[t])
                break
        
        if fail: break

    if fail: break
