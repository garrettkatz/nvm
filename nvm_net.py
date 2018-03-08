import numpy as np
from layer import Layer
from coder import Coder
from gate_map import make_nvm_gate_map
from activator import *
from nvm_instruction_set import flash_instruction_set
from nvm_assembler import assemble

class NVMNet:
    
    def __init__(self, layer_size, pad, devices):

        # set up parameters
        self.layer_size = layer_size
        self.pad = pad

        # set up layers
        act = logistic_activator(pad, layer_size)
        # act = tanh_activator(pad, layer_size)
        layer_names = ['ip','opc','op1','op2','op3']#,'cmph','cmpo']
        layer_names = layer_names[:5]
        layers = {name: Layer(name, layer_size, act, Coder(act)) for name in layer_names}
        layers.update(devices)
        self.layers = layers

        # set up gates
        NL = len(layers) + 2 # +2 for gate out/hidden
        NG = NL**2 + NL # number of gates
        NH = 64 # number of hidden units
        acto = heaviside_activator(NG)
        acth = logistic_activator(pad,NH)
        layers['gate_output'] = Layer('gate_output', NG, acto, Coder(acto))
        layers['gate_hidden'] = Layer('gate_hidden', NH, acth, Coder(acth))
        self.gate_map = make_nvm_gate_map(layers.keys())        

        # setup connection matrices
        weights, biases = flash_instruction_set(self)

        self.weights, self.biases = weights, biases
        
        # initialize layer states
        self.activity = {
            name: layer.activator.off * np.ones((layer.size,1))
            for name, layer in self.layers.items()}
        self.activity['gate_output'] = self.layers['gate_output'].coder.encode('off')
        self.activity['gate_hidden'] = self.layers['gate_hidden'].coder.encode('start')

    def set_pattern(self, layer_name, pattern):
        self.activity[layer_name] = pattern

    def get_open_gates(self):
        pattern = self.activity['gate_output']
        a = self.layers['gate_output'].activator
        open_gates = []
        for k in self.gate_map.gate_keys:
            if self.gate_map.get_gate_value(k, pattern) > a.mid:
                open_gates.append(k)
        return open_gates

    def assemble(self, program, name, learning_rule, verbose=False):
        weights, biases = assemble(self,
            program, name, learning_rule, verbose)
        self.weights.update(weights)
        self.biases.update(biases)

    def tick(self):

        # NVM tick
        current_gates = self.activity['gate_output']
        activity_new = {name: np.zeros(pattern.shape) for name, pattern in self.activity.items()}
        for (to_layer, from_layer) in self.weights:
            u = self.gate_map.get_gate_value((to_layer, from_layer, 'u'), current_gates)
            w = self.weights[(to_layer, from_layer)]
            b = self.biases[(to_layer, from_layer)]
            wvb = u * (w.dot(self.activity[from_layer]) + b)
            activity_new[to_layer] += wvb

        for name, layer in self.layers.items():
            u = self.gate_map.get_gate_value((name, name, 'u'), current_gates)
            d = self.gate_map.get_gate_value((name, name, 'd'), current_gates)
            a = self.activity[name]
            activity_new[name] += (1-u) * (1-d) *  layer.activator.g(a)
            # for tanh this is gain, for logi this is gain and bias

    
        # # handle compare specially, never gated
        # cmp_e = 1./(2.*self.layer_size)
        # w_cmph = np.arctanh(1. - cmp_e) / (PAD / 2.)**2
        # w_cmpo = 2. * np.arctanh(self.pad) / (self.layer_size*(1-cmp_e) - (self.layer_size-1))
        # b_cmpo = w_cmpo * (self.layer_size*(1 - cmp_e) + (self.layer_size-1)) / 2.
        # activity_new['cmph'] = w_cmph * self.activity['CMPA'] * self.activity['CMPB']
        # activity_new['cmpo'] = np.ones((self.layer_size,1)) * (w_cmpo * self.activity['cmph'].sum() - b_cmpo)
        
        for name in activity_new:
            activity_new[name] = self.layers[name].activator.f(activity_new[name])
            # # inject noise
            # flip = (np.random.rand(activity_new[layer].shape[0]) < FLIP_NOISE)
            # activity_new[layer][flip,:] = -activity_new[layer][flip,:]
            # activity_new[layer] += np.random.randn(*activity_new[layer].shape)*CTS_NOISE
        
        self.activity = activity_new

if __name__ == '__main__':

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})
    from learning_rules import logistic_hebbian

    program = """
    
start:  nop
        set r1 true
        mov r2 r1
        jmp cond start
        end
    """
    pname = "test"

    layer_size = 64
    pad = 0.9
    devices = {}

    nvmnet = NVMNet(layer_size, pad, devices)
    nvmnet.assemble(program, pname, logistic_hebbian)
    nvmnet.activity["ip"] = nvmnet.layers["ip"].coder.encode(pname)
    
    show_layers = [
        ["gate_output"],
        ["gate_hidden"],
        ["ip"] + ["op"+x for x in "c123"]]
    for t in range(40):
        # if True:
        # if t % 2 == 0:
        if nvmnet.layers["gate_hidden"].coder.decode(nvmnet.activity["gate_hidden"]) == "start":
            print('t = %d'%t)
            for sl in show_layers:
                print(", ".join(["%s=%s"%(
                    name, nvmnet.layers[name].coder.decode(nvmnet.activity[name]))
                    for name in sl]))
            # print(nvmnet.get_open_gates())
            if nvmnet.layers["opc"].coder.decode(nvmnet.activity["opc"]) == "end":
                break
        nvmnet.tick()
