import numpy as np
from layer import Layer
from coder import Coder
from gate_map import make_nvm_gate_map
from activator import *
from learning_rules import *
from nvm_instruction_set import flash_instruction_set
from nvm_assembler import assemble
from nvm_linker import link

def update_add(accumulator, summand):
    for k, v in summand.items():
        if k in accumulator:
            accumulator[k] += v
        else: accumulator[k] = v

def address_space(forward_layer, backward_layer):
    # set up memory address space
    layers = {'f': forward_layer, 'b': backward_layer}
    N = layers['f'].size
    A = {}
    for d in ['f','b']:
        A[d] = np.concatenate(
            [layers[d].coder.encode(str(a)) for a in range(N)],
            axis=1)
    weights, biases = {}, {}
    for d_to in ['f','b']:
        for d_from in ['f','b']:
            key = (layers[d_to].name, layers[d_from].name)
            Y, X = A[d_to], A[d_from]
            if d_from == 'f': X = np.roll(X, 1, axis=1)
            if d_from == 'b': Y = np.roll(Y, 1, axis=1)
            print("%s residuals:"%str(key))
            weights[key], biases[key], _ = flash_mem(
                np.zeros((N, N)), np.zeros((N, 1)),
                X, Y,
                layers[d_from].activator,
                layers[d_to].activator,
                linear_solve, verbose=True)
    return weights, biases

class NVMNet:
    
    def __init__(self, layer_shape, pad, activator, learning_rule, devices, shapes={}):
        # layer_shape is default, shapes[layer_name] are overrides
        if 'gh' not in shapes: shapes['gh'] = (32,32)
        if 'm' not in shapes: shapes['m'] = (16,16)
        if 's' not in shapes: shapes['s'] = (8,8)
        if 'c' not in shapes: shapes['c'] = (32,32)

        # set up parameters
        self.layer_size = layer_shape[0]*layer_shape[1]
        self.pad = pad

        # set up instruction layers
        act = activator(pad, self.layer_size)
        layer_names = ['ip','opc','op1','op2']
        layers = {name: Layer(name, shapes.get(name, layer_shape), act, Coder(act))
            for name in layer_names}

        # set up memory and stack layers
        NM, NS = shapes['m'][0]*shapes['m'][1], shapes['s'][0]*shapes['s'][1]
        actm, acts = activator(pad, NM), activator(pad, NS)
        for m in ['mf','mb']: layers[m] = Layer(m, shapes['m'], actm, Coder(actm))
        for s in ['sf','sb']: layers[s] = Layer(s, shapes['s'], acts, Coder(acts))

        # set up comparison layers
        NC = shapes['c'][0]*shapes['c'][1]
        actc = activator(pad, NC)
        for c in ['ci','co']: layers[c] = Layer(c, shapes['c'], actc, Coder(actc))
        co_true = layers['co'].coder.encode('true')
        layers['co'].coder.encode('false', np.array([
            [actc.on if tf == actc.off else actc.off]
            for tf in co_true.flatten()]))
        
        # add device layers
        layers.update(devices)
        self.devices = devices
        self.layers = layers

        # set up gain
        self.w_gain, self.b_gain = act.gain()

        # set up gates
        NL = len(layers) + 2 # +2 for gate out/hidden
        NG = NL + 3*NL**2 # number of gates (d + u + l + f)
        NH = shapes['gh'][0]*shapes['gh'][1] # number of hidden units
        acto = heaviside_activator(NG)
        acth = activator(pad,NH)
        layers['go'] = Layer('go', (1,NG), acto, Coder(acto))
        layers['gh'] = Layer('gh', shapes['gh'], acth, Coder(acth))
        self.gate_map = make_nvm_gate_map(layers.keys())        

        # set up connection matrices
        self.weights, self.biases = flash_instruction_set(self)
        for ms in 'ms':
            ms_weights, ms_biases = address_space(
                layers[ms+'f'], layers[ms+'b'])
            self.weights.update(ms_weights)
            self.biases.update(ms_biases)

        # initialize fast connectivity
        connect_pairs = \
            [(device,'mf') for device in self.devices] + \
            [('mf',device) for device in self.devices] + \
            [('ip','sf')] + \
            [('co','ci')]
        for (to_name, from_name) in connect_pairs:
            N_to = self.layers[to_name].size
            N_from = self.layers[from_name].size
            self.weights[(to_name, from_name)] = np.zeros((N_to, N_from))
            self.biases[(to_name, from_name)] = np.zeros((N_to, 1))

        # initialize layer states
        self.activity = {
            name: layer.activator.off * np.ones((layer.size,1))
            for name, layer in self.layers.items()}
        self.activity['go'] = self.layers['go'].coder.encode('start')
        self.activity['gh'] = self.layers['gh'].coder.encode('start')
        for ms in 'ms':
            self.activity[ms+'f'] = self.layers[ms+'f'].coder.encode('0')
            self.activity[ms+'b'] = self.layers[ms+'b'].coder.encode('0')

        # initialize constants
        self.constants = ["null"] #"true", "false"]

        # initialize learning
        self.learning_rules = {
            (to_layer, from_layer): learning_rule
            for to_layer in self.layers for from_layer in self.layers}
        self.learning_rules[('co','ci')] = lambda w, b, x, y, ax, ay: \
            dipole(w, b, x, co_true, ax, ay)

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
        update_add(self.weights, weights)
        update_add(self.biases, biases)

    def link(self, verbose=1, tokens=[], orthogonal=False):
        weights, biases, diff_count = link(
            self, verbose=(verbose > 1), tokens=tokens, orthogonal=orthogonal)
        if verbose > 0: print("linker diff count = %d"%diff_count)
        update_add(self.weights, weights)
        update_add(self.biases, biases)
        return diff_count

    def load(self, program_name, activity):
        # set program pointer
        self.activity["ip"] = self.layers["ip"].coder.encode(program_name)
        # set initial activities
        for layer, token in activity.items():
            self.activity[layer] = self.layers[layer].coder.encode(token)

    def tick(self):

        ### NVM tick
        current_gates = self.activity['go']

        # activity
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

        # plasticity
        for (to_layer, from_layer) in self.weights:
            l = self.gate_map.get_gate_value(
                (to_layer, from_layer, 'l'), current_gates)
            f = self.gate_map.get_gate_value(
                (to_layer, from_layer, 'f'), current_gates)
            pair_key = (to_layer, from_layer)

            # actg = self.layers["go"].activator
            # if np.fabs(l - actg.on) < np.fabs(l - actg.off):
            # if True:
            if f != 0: # forget before learning since uses previous weights

                old_to = self.layers[to_layer].activator.f(
                    self.weights[pair_key].dot(self.activity[from_layer]) +
                    self.biases[pair_key])

                dw, db = self.learning_rules[pair_key](
                    self.weights[pair_key],
                    self.biases[pair_key],
                    self.activity[from_layer],
                    old_to,
                    self.layers[from_layer].activator,
                    self.layers[to_layer].activator)

                self.weights[pair_key] -= f*dw
                self.biases[pair_key] -= f*db

            if l != 0:

                dw, db = self.learning_rules[pair_key](
                    self.weights[pair_key],
                    self.biases[pair_key],
                    self.activity[from_layer],
                    self.activity[to_layer],
                    self.layers[from_layer].activator,
                    self.layers[to_layer].activator)

                self.weights[pair_key] += l*dw
                self.biases[pair_key] += l*db

        self.activity = activity_new

    def at_start(self):
        return self.layers["gh"].coder.decode(self.activity["gh"]) == "start"

    def at_exit(self):
        return (self.layers["opc"].coder.decode(self.activity["opc"]) == "exit")

    def state_string(self, show_layers=[], show_tokens=False, show_corrosion=False, show_gates=False):
        s = ""
        for sl in show_layers:
            if show_tokens:
                s += ", ".join(["%s=%s"%(
                    name, self.layers[name].coder.decode(
                        self.activity[name]))
                    for name in sl])
                s += "\n"
            if show_corrosion:
                s += ", ".join(["%s~%.2f"%(
                    name, self.layers[name].activator.corrosion(
                        self.activity[name]))
                    for name in sl])
                s += "\n"
        if show_gates:
            s += str(self.get_open_gates())
            s += "\n"
        return s

def make_nvmnet(programs=None, devices=None):

    # default program
    if programs is None:
        programs = {"test":"""
    
                mov d2 true
        loop:   mov d1 here
                jmp d1
        here:   mov d0 d2
                exit
    
        """}

    # set up activator
    activator, learning_rule = logistic_activator, hebbian
    # activator, learning_rule = tanh_activator, hebbian

    # make network
    layer_shape = (16, 16)
    layer_size = layer_shape[0]*layer_shape[1]
    pad = 0.01
    act = activator(pad, layer_size)

    # default devices
    if devices is None:
        devices = {"d%d"%d: Layer("d%d"%d, layer_shape, act, Coder(act))
            for d in range(3)}

    # assemble and link programs
    nvmnet = NVMNet(layer_shape, pad, activator, learning_rule, devices)
    for name, program in programs.items():
        nvmnet.assemble(program, name, verbose=1)
    nvmnet.link(verbose=1)

    # initialize pointer at last program
    nvmnet.activity["ip"] = nvmnet.layers["ip"].coder.encode(name)

    return nvmnet

if __name__ == '__main__':

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})
   
    nvmnet = make_nvmnet()
    raw_input("continue?")
        
    show_layers = [
        ["go", "gh","ip"] + ["op"+x for x in "c12"] + ["d0","d1","d2"],
    ]
    show_tokens = True
    show_corrosion = True
    show_gates = False

    for t in range(100):
        # if True:
        # if t % 2 == 0 or at_exit:
        if nvmnet.at_start() or nvmnet.at_exit():
            print('t = %d'%t)
            print(nvmnet.state_string(show_layers, show_tokens, show_corrosion, show_gates))
        if nvmnet.at_exit():
            break
        nvmnet.tick()
