from activator import *
from learning_rules import *
from layer import Layer
from coder import Coder
from nvm_net import NVMNet
from preprocessing import *
from orthogonal_patterns import nearest_valid_hadamard_size

class NVM:
    def __init__(self, layer_shape, pad, activator, learning_rule, register_names, shapes={}, tokens=[], orthogonal=False, verbose=False):

        self.tokens = tokens
        self.orthogonal = orthogonal
        self.register_names = register_names
        # default registers
        layer_size = layer_shape[0]*layer_shape[1]
        act = activator(pad, layer_size)
        registers = {name: Layer(name, layer_shape, act, Coder(act))
            for name in register_names}
        self.net = NVMNet(layer_shape, pad, activator, learning_rule, registers, shapes=shapes, tokens=tokens, orthogonal=orthogonal, verbose=verbose)

    def assemble(self, programs, verbose=0, other_tokens=[]):
        self.net.assemble(programs, verbose, self.orthogonal, self.tokens.union(other_tokens))

    def load(self, program_name, initial_state):
        self.net.load(program_name, initial_state)

    def initialize_memory(self, pointers, values):
        self.net.initialize_memory(pointers, values)

    def decode_layer(self, layer_name):
        return self.net.layers[layer_name].coder.decode(self.net.activity[layer_name])

    def encode_symbol(self, layer_name, symbol):
        pattern = self.net.layers[layer_name].coder.encode(symbol)
        self.net.activity[layer_name] = pattern

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
                "%s:%s"%(r,state[r]) for r in self.net.registers])

    def at_start(self):
        return self.net.at_start()

    def at_exit(self):
        return self.net.at_exit()

    def step(self, verbose=0, max_ticks=50):
        for t in range(max_ticks):
            self.net.tick()

            if verbose > 1: print(self.state_string())
            elif self.net.at_ready() and verbose > 0: print(self.state_string())

            if self.net.at_start(): return True
            if self.at_exit(): return True

        # indicate whether step failed
        return False

def make_default_nvm(register_names, layer_shape=None, orthogonal=False, shapes={}, tokens=[]):
    # if layer_shape is None: layer_shape = (16,16) if orthogonal else (32,32)
    if layer_shape is None: layer_shape = (12,20) if orthogonal else (32,32) # test non-pow-2 hadamard
    pad = 0.0001
    # activator, learning_rule = tanh_activator, rehebbian
    activator, learning_rule = logistic_activator, rehebbian

    return NVM(layer_shape,
        pad, activator, learning_rule, register_names,
        shapes=shapes, tokens=tokens, orthogonal=orthogonal)

def make_scaled_nvm(register_names, programs, orthogonal=False, capacity_factor=.05, scale_factor=1.0, extra_tokens=[], num_addresses=None, verbose=False):
    """
    Create an NVM with auto-scaled layer sizes based on programs that will be learned
    capacity_factor: assumes pattern capacity is at most this fraction of layer size
    scale_factor: scales layer sizes to this amount of what target capacity requires
    """
    
    num_lines, num_patterns, all_tokens = measure_programs(
        programs, register_names, extra_tokens=extra_tokens)
    
    layer_size = int(nearest_valid_hadamard_size(scale_factor * num_patterns)
        if orthogonal else scale_factor * num_patterns / capacity_factor)
    ip_size = int(nearest_valid_hadamard_size(scale_factor * (num_lines+1)) # +1 for program name
        if orthogonal else scale_factor * num_lines/capacity_factor)
    
    layer_shape = (layer_size, 1)
    shapes = {'ip': (ip_size, 1)}

    if num_addresses is not None:
        m_size = int(nearest_valid_hadamard_size(scale_factor * num_addresses)
        if orthogonal else scale_factor * num_addresses/capacity_factor)
        shapes['m'] = (m_size,1)

    # avoid non-deterministic transits with very small layer_size
    for layer_name in ["opc", "op1", "op2"]:
        if layer_name not in shapes:
            shapes[layer_name] = (max(layer_size, 16), 1)

    pad = 0.0001
    activator, learning_rule = tanh_activator, rehebbian
    # activator, learning_rule = logistic_activator, rehebbian

    return NVM(layer_shape,
        pad, activator, learning_rule, register_names,
        shapes=shapes, tokens=all_tokens, orthogonal=orthogonal, verbose=verbose)

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
