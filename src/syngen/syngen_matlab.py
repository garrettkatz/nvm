import sys
sys.path.append('../nvm')

from nvm import NVM
from activator import tanh_activator
from learning_rules import rehebbian

from syngen_nvm import *
from syngen import Network, Environment
from syngen import get_cpu, get_gpus, interrupt_engine

import numpy as np
from random import random, sample, choice

from threading import Thread

thread = None
global_in = None
global_out = None

def launch():
    global thread

    sys.stdout = sys.__stdout__
    thread = Thread(target=run)
    thread.start()

def kill():
    global thread
    interrupt_engine()
    thread.join()
    thread = None

def insert(x):
    global global_in
    while global_in is not None: pass
    global_in = x

def get():
    global global_out
    while global_out is None: pass
    x = global_out
    global_out = None
    return x

def run():
    verbose = True

    numerals = [str(x) for x in range(0,10)]
    all_tokens = numerals + ["read", "write", "null"]

    values = {}
    pointers = {"0": {"r0": "ptr"}}
    memory = (pointers, values)

    program = """
    start:  mov r0 ptr
            drf r0

    input:  mov r0 read

    wait:   cmp r0 read
            jie wait

            mem r1
            nxt

            cmp r1 null
            jie output
            jmp input

    output: mov r0 ptr
            drf r0

    loop:   rem r1
            mov r0 write
            nxt

    wait2:  cmp r0 write
            jie wait2

            cmp r1 null
            jie end
            jmp loop

    end:    exit
    """

    programs = {"test":program}
    tokens = all_tokens + ["ptr"]
    num_registers = 2

    orthogonal = True
    layer_shape = (16,16) if orthogonal else (32,32)
    pad = 0.0001
    activator, learning_rule = tanh_activator, rehebbian
    register_names = ["r%d"%r for r in range(num_registers)]

    vm = NVM(layer_shape,
        pad, activator, learning_rule, register_names,
        shapes={}, tokens=tokens, orthogonal=orthogonal)

    if memory is not None: vm.initialize_memory(*memory)
    vm.assemble(programs, verbose=0)
    vm.load("test", {})

    syn_net = SyngenNVM(vm.net)

    ### BUILD ENVIRONMENT ####
    output_layers = vm.net.layers.keys() if verbose else []

    syn_env = SyngenEnvironment()

    # Feed inputs from global_in
    def producer():
        global global_in

        if global_in is not None:
            sym = global_in
            global_in = None
            print("Produced %s" % sym)
            return True, sym
        else:
            return False, None

    # Push outputs to global_out
    def consumer(output):
        global global_out

        if global_out is None:
            global_out = output
            print("Consumed %s" % output)
            return True
        else:
            return False

    syn_env.add_streams(vm.net, "r0", "r1", producer, consumer)

    #syn_env.add_visualizer("nvm", output_layers)
    #syn_env.add_printer(vm.net, output_layers)

    ### RUN SYNGEN ENGINE ###
    report = syn_net.run(syn_env, {
        "multithreaded" : True,
        "worker threads" : 0,
        "verbose" : False})

    if verbose:
        print(report)

    syn_net.free()

if __name__ == "__main__":
    main()
