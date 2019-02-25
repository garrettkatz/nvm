import sys
sys.path.append('../nvm')

from sequencer import Sequencer
from gate_sequencer import GateSequencer
from gate_map import GateMap
from activator import heaviside_activator, tanh_activator
from layer import Layer
from coder import Coder
from syngen_nvm import *

import numpy as np
from itertools import product

arith_ops = {
    "+" : (lambda x,y:str( (int(x) + int(y)) / 10 ),
           lambda x,y:str( (int(x) + int(y)) % 10 )),
    "-" : (lambda x,y:str( (int(x) - int(y)) / 10 ),
           lambda x,y:str( (int(x) - int(y)) % 10 )),
    "*" : (lambda x,y:str( (int(x) * int(y)) / 10 ),
           lambda x,y:str( (int(x) * int(y)) % 10 )),
    "/" : (lambda x,y:str( (int(x) % int(y)) ),
           lambda x,y:str( (int(x) / int(y)) )),
}

unary_ops = {
    "++" : (lambda x:str( (int(x)+1) % 10 ), ),
    "--" : (lambda x:str( (int(x)-1) % 10 ), ),
}

comp_ops = {
    "<"  : (lambda x,y:str( int(x) <  int(y) ).lower(), ),
    ">"  : (lambda x,y:str( int(x) >  int(y) ).lower(), ),
    "<=" : (lambda x,y:str( int(x) <= int(y) ).lower(), ),
    ">=" : (lambda x,y:str( int(x) >= int(y) ).lower(), ),
}


class OpDef:
    def __init__(self, op_name, operations, in_ops, out_ops):
        self.op_name = op_name
        self.operations = dict(operations)
        self.in_ops = list(in_ops)
        self.out_ops = list(out_ops)
        if 'null' not in out_ops:
            self.out_ops.append('null')
        self.tokens = list(set(
            list(self.operations.keys()) + self.in_ops + self.out_ops))

def make_arith_opdef(in_range=range(0,10),out_range=range(-9,10)):
    return OpDef("arith", arith_ops,
        (str(x) for x in in_range),
        (str(x) for x in out_range))

def make_unary_opdef(in_range=range(0,10),out_range=range(-9,10)):
    return OpDef("unary", unary_ops,
        (str(x) for x in in_range),
        (str(x) for x in out_range))

def make_comp_opdef(in_range=range(0,10)):
    return OpDef("comp", comp_ops,
        (str(x) for x in in_range),
        ("true", "false"))

class OpNet:
    def __init__(self, nvmnet, opdef, arg_regs, res_regs, op_reg):
        self.opdef = opdef
        self.op_name = opdef.op_name
        self.operations = dict(opdef.operations)
        self.in_ops = list(opdef.in_ops)
        self.out_ops = list(opdef.out_ops)
        self.tokens = list(opdef.tokens)

        self.arg_registers = arg_regs
        self.res_registers = res_regs
        self.op_register = op_reg

        self.hidden_name = "%s_gh" % self.op_name
        self.gate_name = "%s_go" % self.op_name

        # 1. OP->HID, 2. OP->OP, [3. RESX->RESX, 4. RESX->RESY for op in ops]
        self.gate_map = GateMap(
            [(self.hidden_name, self.op_register, "u"),
             (self.op_register, self.op_register, "u")]
            + [("res","res",op) for op in self.operations]
            + [("res","arg",op) for op in self.operations])

        # Hidden gate layer
        N = 16
        self.hidden_size = N**2
        hid_activator = tanh_activator(nvmnet.pad, self.hidden_size)
        self.hidden_layer = Layer(self.hidden_name,
            (N,N), hid_activator, Coder(hid_activator))

        # Gate output layer
        self.gate_size = self.gate_map.get_gate_count()
        gate_activator = heaviside_activator(self.gate_size)
        self.gate_layer = Layer(self.gate_name,
            (self.gate_size,1), gate_activator, Coder(gate_activator))

        # Gate layer (detects operator)
        hidden_gate_layer = {
            "name" : self.hidden_name,
            "neural model" : "nvm",
            "rows" : 1,
            "columns" : self.hidden_size,
        }
        gate_layer = {
            "name" : self.gate_name,
            "neural model" : "nvm_heaviside",
            "rows" : 1,
            "columns" : self.gate_size,
        }
        self.structure = {
            "name" : self.op_name,
            "type" : "parallel",
            "layers": [hidden_gate_layer, gate_layer]
        }

        # Make gate connection
        def build_gate(to_name, index, suffix=""):
            return {
                "name" : get_conn_name(to_name, self.gate_name, suffix),
                "from layer" : self.gate_name,
                "to layer" : to_name,
                "type" : "subset",
                "opcode" : "add",
                "subset config" : {
                    "from row end" : 1,
                    "from column start" : index,
                    "from column end" : index+1,
                    "to row end" : 1,
                    "to column end" : 1,
                },
                "plastic" : False,
                "gate" : True,
            }

        # Squash weights to cancel gain
        def build_squash(to_name, suffix="", gated=True):
            return {
                "name" : get_conn_name(to_name, to_name, suffix),
                "from layer" : "bias",
                "to layer" : to_name,
                "type" : "fully connected",
                "opcode" : "add",
                "plastic" : False,
                "gated" : gated,
            }

        # Make weight/bias connections
        def build_conns(to_name, from_name, suffix="", gated=True):
            return [{
                "name" : get_conn_name(to_name, from_name, suffix + "-w"),
                "from layer" : from_name,
                "to layer" : to_name,
                "type" : "fully connected",
                "opcode" : "add",
                "plastic" : False,
                "gated" : gated
            },
            {
                "name" : get_conn_name(to_name, from_name, suffix + "-b"),
                "from layer" : 'bias',
                "to layer" : to_name,
                "type" : "fully connected",
                "opcode" : "add",
                "plastic" : False,
                "gated" : gated
            }]

        self.connections = []

        # Hidden gate input
        self.connections.append(
            build_gate(self.hidden_name,
                self.gate_map.get_gate_index(
                    (self.hidden_name, self.op_register, "u"))))
        self.connections += build_conns(
            self.hidden_name, self.op_register, gated=True)

        # Hidden gate recurrence
        self.connections += build_conns(
            self.hidden_name, self.hidden_name, gated=False)

        # Gate activation
        self.connections += build_conns(
            self.gate_name, self.hidden_name, gated=False)

        # Operation squash
        self.connections.append(
            build_gate(self.op_register,
                self.gate_map.get_gate_index(
                    (self.op_register, self.op_register, "u")),
                self.op_name))
        self.connections.append(build_squash(
            self.op_register, suffix=self.op_name+"-squash"))

        for op in self.operations:
            for to_name in self.res_registers:
                # Recurrent connections
                self.connections.append(
                    build_gate(to_name,
                        self.gate_map.get_gate_index(
                            ("res", "res", op)),
                        op+"-1"))
                self.connections += build_conns(
                    to_name, to_name, suffix=op, gated=True)

                # Inter-layer connections
                self.connections.append(
                    build_gate(to_name,
                        self.gate_map.get_gate_index(
                            ("res", "arg", op)),
                        op+"-2"))
                for from_name in self.arg_registers:
                    if to_name != from_name:
                        self.connections += build_conns(
                            to_name, from_name, suffix=op, gated=True)

        self.layer_map = { name : nvmnet.layers[name] for name in
            self.arg_registers + self.res_registers + [self.op_register] }
        self.layer_map[self.gate_name] = self.gate_layer
        self.layer_map[self.hidden_name] = self.hidden_layer

        self.conn_names = tuple(conn["name"] for conn in self.connections)



    def initialize_weights(self, syngen_net, nvmnet):

        def set_seq_weights(ws, bs, suffix=""):
            for to_name, from_name in ws:
                w = ws[(to_name,from_name)]
                b = bs[(to_name,from_name)]

                # Recurrent connections need to overcome gain
                # TODO: Eventually, this should be resolved by gating
                #     (to_name, to_name, 'd')
                if to_name == from_name:
                    try: w = w + (-nvmnet.w_gain[to_name] * np.eye(w.shape[0]))
                    except KeyError: pass

                w_name = get_conn_name(to_name, from_name, suffix + "-w")
                if w_name in self.conn_names:
                    syngen_net.net.get_weight_matrix(w_name).copy_from(w.flat)

                b_name = get_conn_name(to_name, from_name, suffix + "-b")
                if b_name in self.conn_names:
                    syngen_net.net.get_weight_matrix(b_name).copy_from(b.flat)

        op_layer = nvmnet.layers[self.op_register]

        # Hidden gate transits
        hid_seq = GateSequencer(
            self.gate_map, self.gate_layer, self.hidden_layer,
            { self.hidden_name: self.hidden_layer, self.op_register: op_layer },
            default_gates=[])

        # Start state recurrence
        v_start = self.hidden_layer.coder.encode("START")
        v_end = self.hidden_layer.coder.encode("END")
        self.hidden_start_state = v_start
        self.hidden_end_state = v_end

        # START -> END
        hid_seq.add_transit(new_hidden = v_end, old_hidden = v_start)

        # END -> START
        # OP -> Hidden Gate
        load_gates,_ = hid_seq.add_transit(
            new_hidden = v_start, old_hidden = v_end,
            ungate=[(self.hidden_name, self.op_register, "u")])

        for op in self.operations.keys():
            # Squash OP + RES Recurrence + Inter-RES
            hid_seq.add_transit(new_hidden = op, old_hidden = v_start,
                ungate = [
                    (self.op_register, self.op_register, "u"),
                    ("res", "res", op),
                    ("res", "arg", op)],
                old_gates=load_gates, **{ self.op_register : op})

            # RES Recurrence
            hid_seq.add_transit(new_hidden = v_end, old_hidden = op,
                intermediate_ungate = [("res", "res", op)])

        ws, bs,  _ = hid_seq.flash(False)
        set_seq_weights(ws, bs)


        # OP squash
        w = op_layer.activator.g(op_layer.coder.encode("null")) * 10
        syngen_net.net.get_weight_matrix(
            get_conn_name(self.op_register, self.op_register,
                self.op_name + "-squash")).copy_from(w.flat)

        # Result transits
        for op,fs in self.operations.items():

            input_layers = { name : nvmnet.layers[name]
                for name in self.arg_registers + self.res_registers }
            seqs = [ Sequencer(nvmnet.layers[name], input_layers)
                for name in self.res_registers ]

            # Create the map
            combos = product(self.in_ops, repeat=len(self.arg_registers))
            for elems in combos:
                input_states = dict(zip(self.arg_registers, elems))

                # Ensure that res_layers are represented in the input
                for res_layer in self.res_registers:
                    if res_layer not in input_states:
                        input_states[res_layer] = np.zeros(
                            (nvmnet.layers[res_layer].size,1))

                try:
                    outputs = tuple(f(*elems) for f in fs)
                except:
                    outputs = tuple('null' for f in fs)

                for name,output,seq in zip(self.res_registers, outputs, seqs):
                    seq.add_transit(output, **input_states)

            for seq in seqs:
                ws, bs, _, _ = seq.flash(False)
                set_seq_weights(ws, bs, op)


    def initialize_activity(self, syngen_net, nvmnet):
        # Initial hidden state
        syngen_net.net.get_neuron_data(
            self.op_name, self.hidden_name, "output").copy_from(
                self.hidden_start_state.flat)
