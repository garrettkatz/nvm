import numpy as np
from nvm import make_default_nvm
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
    print(subroutine_partition)

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
    layer_shape,
    orthogonal,
    program,
    tokens,
    initial_activity,
    max_steps,
    verbose=0):

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
    
    return leading_match_count, trial_step_count, nvm_trace, rvm_trace
    
if __name__ == "__main__":

    max_steps = 20
    num_tokens = 2
    num_subroutines = 2
    num_lines = 20
    num_registers = 2
    register_names = ["r%d"%r for r in range(num_registers)]

    # Generate program
    program, tokens, initial_activity = generate_random_program(
        register_names, num_tokens, num_subroutines, num_lines)

    layer_shape = (16,8)
    orthogonal=True
    leading_match_count, trial_step_count, nvm_trace, rvm_trace = match_trial(
        register_names, layer_shape, orthogonal,
        program, tokens, initial_activity,
        max_steps, verbose=0)

    for t in nvm_trace: print(t)
    print(program)
    for t in rvm_trace: print(t)
    print("matched leading %d of %d"%(leading_match_count, trial_step_count))
