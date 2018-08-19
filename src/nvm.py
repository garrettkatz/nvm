"""
Symbolically implemented reference machine
"""
from activator import *
from learning_rules import *
from layer import Layer
from coder import Coder
from nvm_net import NVMNet
from preprocessing import preprocess
from orthogonal_patterns import nearest_power_of_2

class NVM:
    def __init__(self, layer_shape, pad, activator, learning_rule, register_names, shapes={}, tokens=[], orthogonal=False):

        self.tokens = tokens
        self.orthogonal = orthogonal
        self.register_names = register_names
        # default registers
        layer_size = layer_shape[0]*layer_shape[1]
        act = activator(pad, layer_size)
        registers = {name: Layer(name, layer_shape, act, Coder(act))
            for name in register_names}
        self.net = NVMNet(layer_shape, pad, activator, learning_rule, registers, shapes=shapes, tokens=tokens, orthogonal=orthogonal)

    def assemble(self, programs, verbose=1, other_tokens=[]):
        self.net.assemble(programs, verbose, self.orthogonal, other_tokens)

    def load(self, program_name, initial_state):
        self.net.load(program_name, initial_state)

    def initialize_memory(self, pointers, values):
        self.net.initialize_memory(pointers, values)

    def decode_state(self, layer_names=None):
        if layer_names is None:
            layer_names = self.net.layers.keys()
        return {name:
            self.net.layers[name].coder.decode(
            self.net.activity[name])
            for name in layer_names}

    def state_string(self):
        state = self.decode_state()
        return "ip %s: "%state["ip"] + \
            " ".join([
                "%s"%state[x] for x in ["opc","op1","op2"]]) + ", " + \
            ",".join([
                "%s:%s"%(r,state[r]) for r in self.net.devices])


    def at_exit(self):
        return self.net.at_exit()

    def step(self, verbose=0, max_ticks=50):
        for t in range(max_ticks):
            self.net.tick()

            if verbose > 1: print(self.state_string())
            elif self.net.at_ready() and verbose > 0: print(self.state_string())

            if self.net.at_start(): break
            if self.at_exit(): break

def make_default_nvm(register_names, layer_shape=None, orthogonal=False, shapes={}, tokens=[]):
    if layer_shape is None: layer_shape = (16,16) if orthogonal else (32,32)
    pad = 0.0001
    # activator, learning_rule = tanh_activator, rehebbian
    activator, learning_rule = logistic_activator, rehebbian

    return NVM(layer_shape,
        pad, activator, learning_rule, register_names,
        shapes=shapes, tokens=tokens, orthogonal=orthogonal)

def make_scaled_nvm(register_names, programs, orthogonal=False, capacity_factor=.138, scale_factor=1.0, tokens=[]):
    """
    Create an NVM with auto-scaled layer sizes based on programs that will be learned
    capacity_factor: assumes pattern capacity is at most this fraction of layer size
    scale_factor: scales layer sizes to this amount of what target capacity requires
    """
    
    lines, labels, all_tokens = preprocess(programs, register_names)
    all_tokens |= set(tokens)
    num_tokens = len(all_tokens)
    num_lines = sum([len(lines[name]) for name in lines])
    
    layer_shape = (
        int(nearest_power_of_2(scale_factor * num_tokens) if orthogonal else
        scale_factor * num_tokens / capacity_factor),
        1)
    shapes = {'ip': (
        int(nearest_power_of_2(scale_factor * num_lines) if orthogonal else
        scale_factor * num_lines/capacity_factor),
        1)}

    pad = 0.0001
    activator, learning_rule = tanh_activator, rehebbian
    # activator, learning_rule = logistic_activator, rehebbian

    return NVM(layer_shape,
        pad, activator, learning_rule, register_names,
        shapes=shapes, tokens=all_tokens, orthogonal=orthogonal)    

if __name__ == "__main__":

    programs = {"test":"""

    # start:  nop
    start:  mov r1 A
            jmpv jump
            nop
    jump:   jie end
            mov r2 r1
            nop
    end:    exit
    
    """}

    # layer_shape = (32,32)
    # pad = 0.0001
    # activator, learning_rule = logistic_activator, hebbian
    # # activator, learning_rule = tanh_activator, hebbian

    # nvm = NVM(layer_shape, pad, activator, learning_rule, register_names=["r%d"%r for r in range(3)], shapes={}, tokens=[], orthogonal=False)
    
    register_names = ["r%d"%r for r in range(3)]
    nvm = make_scaled_nvm(register_names, programs, orthogonal=True, capacity_factor=.138, scale_factor=1.0, tokens=["start","jump","end","A","B","C"])
    
    nvm.assemble(programs, verbose=1)
    nvm.load("test",{"r0":"A","r1":"B","r2":"C"})

    for t in range(10):
        print(nvm.state_string())
        nvm.step()
        if nvm.net.at_exit(): break
