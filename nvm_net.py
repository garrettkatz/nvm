import numpy as np
from layer import Layer
from coder import Coder
from gate_map import make_nvm_gate_map
from activator import *
from learning_rules import *
from nvm_instruction_set import flash_instruction_set
from nvm_assembler import assemble
from nvm_linker import link

class NVMNet:
    
    def __init__(self, layer_size, pad, activator, learning_rule, devices):

        # set up parameters
        self.layer_size = layer_size
        self.pad = pad
        self.learning_rule = learning_rule

        # set up layers
        act = activator(pad, layer_size)
        layer_names = ['ip','opc','op1','op2','op3']#,'cmph','cmpo']
        layer_names = layer_names[:5]
        layers = {name: Layer(name, layer_size, act, Coder(act)) for name in layer_names}
        layers.update(devices)
        self.devices = devices
        self.layers = layers

        # set up gain
        self.w_gain, self.b_gain = act.gain()

        # set up gates
        NL = len(layers) + 2 # +2 for gate out/hidden
        NG = NL**2 + NL # number of gates
        NH = 256 # number of hidden units
        acto = heaviside_activator(NG)
        acth = activator(pad,NH)
        layers['go'] = Layer('go', NG, acto, Coder(acto))
        layers['gh'] = Layer('gh', NH, acth, Coder(acth))
        self.gate_map = make_nvm_gate_map(layers.keys())        

        # setup connection matrices
        weights, biases = flash_instruction_set(self)

        self.weights, self.biases = weights, biases
        
        # initialize layer states
        self.activity = {
            name: layer.activator.off * np.ones((layer.size,1))
            for name, layer in self.layers.items()}
        self.activity['go'] = self.layers['go'].coder.encode('start')
        self.activity['gh'] = self.layers['gh'].coder.encode('start')

        # initialize constants
        self.constants = ["true", "false", "null"]

    def set_pattern(self, layer_name, pattern):
        self.activity[layer_name] = pattern

    def get_open_gates(self):
        pattern = self.activity['go']
        a = self.layers['go'].activator
        open_gates = []
        for k in self.gate_map.gate_keys:
            g = self.gate_map.get_gate_value(k, pattern)
            if np.fabs(g - a.on) < np.fabs(g - a.off):
                open_gates.append(k)
        return open_gates

    def assemble(self, program, name, verbose=1):
        weights, biases, diff_count = assemble(self,
            program, name, verbose=(verbose > 1))
        if verbose > 0: print("assembler diff count = %d"%diff_count)
        self.weights.update(weights)
        self.biases.update(biases)

    def link(self, verbose=1):
        weights, biases, diff_count = link(self, verbose=(verbose > 1))
        self.weights.update(weights)
        self.biases.update(biases)
        if verbose > 0: print("linker diff count = %d"%diff_count)

    def tick(self):

        # NVM tick
        current_gates = self.activity['go']
        activity_new = {name: np.zeros(pattern.shape)
            for name, pattern in self.activity.items()}
        
        for (to_layer, from_layer) in self.weights:
            u = self.gate_map.get_gate_value(
                (to_layer, from_layer, 'u'), current_gates)
            w = self.weights[(to_layer, from_layer)]
            b = self.biases[(to_layer, from_layer)]
            wvb = u * (w.dot(self.activity[from_layer]) + b)
            activity_new[to_layer] += wvb

        for name, layer in self.layers.items():
            u = self.gate_map.get_gate_value((name, name, 'u'), current_gates)
            d = self.gate_map.get_gate_value((name, name, 'd'), current_gates)
            wvb = self.w_gain * self.activity[name] + self.b_gain
            activity_new[name] += (1-u) * (1-d) * wvb
    
        for name in activity_new:
            activity_new[name] = self.layers[name].activator.f(activity_new[name])
        
        self.activity = activity_new

def make_nvmnet():

    program = """

    loop:   set d1 true
            mov d2 d1
            jif d2 start
            exit

    """
    pname = "test"

    activator, learning_rule = logistic_activator, logistic_hebbian
    # activator, learning_rule = tanh_activator, tanh_hebbian

    layer_size = 128
    pad = 0.025
    act = activator(pad, layer_size)
    devices = {"d%d"%d: Layer("d%d"%d, layer_size, act, Coder(act))
        for d in range(3)}

    nvmnet = NVMNet(layer_size, pad, activator, learning_rule, devices)
    nvmnet.assemble(program, pname, verbose=1)
    nvmnet.link(verbose=1)
    nvmnet.activity["ip"] = nvmnet.layers["ip"].coder.encode(pname)

    return nvmnet

if __name__ == '__main__':

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})
   
    nvmnet = make_nvmnet()
    raw_input("continue?")
        
    show_layers = [
        ["go", "gh","ip"] + ["op"+x for x in "c123"] + ["d1","d2"],
    ]
    for t in range(60):
        at_start = nvmnet.layers["gh"].coder.decode(nvmnet.activity["gh"]) == "start"
        at_exit = nvmnet.layers["opc"].coder.decode(nvmnet.activity["opc"]) == "exit"
        # if True:
        # if t % 2 == 0 or at_exit:
        if at_start or at_exit:
            print('t = %d'%t)
            for sl in show_layers:
                print(", ".join(["%s=%s"%(
                    name, nvmnet.layers[name].coder.decode(nvmnet.activity[name]))
                    for name in sl]))
                # print(", ".join(["%s~%.2f"%(
                #     name,
                #     nvmnet.layers[name].activator.corrosion(nvmnet.activity[name]))
                #     for name in sl]))
            # print(nvmnet.get_open_gates())
        if at_exit:
            break
        nvmnet.tick()
