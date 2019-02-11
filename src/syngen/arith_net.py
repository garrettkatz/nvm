import sys
sys.path.append('../nvm')

from sequencer import Sequencer
from syngen_nvm import *

import numpy as np

operations = {
    "+" : (lambda x,y:(x+y)/10, lambda x,y:(x+y)%10),
    "-" : (lambda x,y:(x-y)/10, lambda x,y:(x-y)%10),
    "*" : (lambda x,y:(x*y)/10, lambda x,y:(x*y)%10),
    "/" : (lambda x,y:(x%y),    lambda x,y:(x/y)),
}

class ArithNet:
    def __init__(self, nvmnet, x_name="r0", y_name="r1", op_name="r2"):
        self.x_name = x_name
        self.y_name = y_name
        self.op_name = op_name

        op_reg = nvmnet.layers[op_name]
        gate_bias = op_reg.activator.on * -(op_reg.size - 1)

        # Create detector for operators
        arith_gate = {
            "name" : "arith_gate",
            "neural model" : "binary threshold",
            "rows" : 1,
            "columns" : len(operations),
            "init config": {
                "type": "flat",
                "value": gate_bias
            }
        }
        self.structure = {
            "name" : "arith",
            "type" : "parallel",
            "layers": [arith_gate]
        }


        self.connections = []

        # Arith gate activation (detect operation)
        self.connections.append({
            "name" : "arith_gate<%s" % op_name,
            "from layer" : op_name,
            "to layer" : 'arith_gate',
            "type" : "fully connected",
            "opcode" : "add",
            "plastic" : False,
        })

        # Make gate connection
        def make_gate(name, suffix, start_index, end_index):
            return {
                "name" : "%s<arith_gate%s" % (name, suffix),
                "from layer" : "arith_gate",
                "to layer" : name,
                "type" : "subset",
                "subset config" : {
                    "from row end" : 1,
                    "from column start" : start_index,
                    "from column end" : end_index,
                    "to row end" : 1,
                    "to column end" : 1,
                },
                "plastic" : False,
                "gate" : True
            }

        # Make weight/bias connections
        def make_conns(to_name, from_name, suffix):
            return [{
                "name" : get_conn_name(to_name, from_name, '-arith_w' + suffix),
                "from layer" : from_name,
                "to layer" : to_name,
                "type" : "fully connected",
                "opcode" : "add",
                "plastic" : False,
                "gated" : True
            },
            {
                "name" : get_conn_name(to_name, from_name, '-arith_b' + suffix),
                "from layer" : 'bias',
                "to layer" : to_name,
                "type" : "fully connected",
                "opcode" : "add",
                "plastic" : False,
                "gated" : True
            }]

        # operation arith gate/conn (operation -> null)
        self.connections.append(make_gate(op_name, "", 0, len(operations)))
        self.connections += make_conns(op_name, op_name, "")

        # x/y operand gates/connections
        for index,op in enumerate(operations.keys()):
            for to_name in [x_name, y_name]:
                self.connections.append(
                    make_gate(to_name, "_%s" % op, index, index+1))

                for from_name in [x_name, y_name]:
                    self.connections += make_conns(
                        to_name, from_name, "_%s" % op)


    def initialize(self, syngen_net, nvmnet):

        def flash(seq, suffix=""):
            to_name = seq.sequence_layer.name
            ws, bs, _, _ = seq.flash(False)

            for from_name in seq.input_layers.keys():
                w = ws[(to_name,from_name)]
                b = bs[(to_name,from_name)]

                if to_name == from_name:
                    w = w + (-nvmnet.w_gain[to_name] * np.eye(w.shape[0]))

                syngen_net.net.get_weight_matrix(get_conn_name(
                    to_name, from_name, '-arith_w' + suffix)).copy_from(w.flat)
                syngen_net.net.get_weight_matrix(get_conn_name(
                    to_name, from_name, '-arith_b' + suffix)).copy_from(b.flat)

        # Detector weights
        op_layer = nvmnet.layers[self.op_name]
        w = np.zeros((0,op_layer.size))
        for op in operations.keys():
            op_pattern = np.sign(
                op_layer.coder.encode(op).reshape((1,op_layer.size)))
            w = np.append(w, op_pattern, axis=0)
        syngen_net.net.get_weight_matrix(
            "arith_gate<%s" % self.op_name).copy_from(w.flat)

        # operator transits (squash +, map to null)
        op_seq = Sequencer(nvmnet.layers[self.op_name],
            { self.op_name : nvmnet.layers[self.op_name] })
        for op in operations.keys():
            op_seq.add_transit('null', **{ self.op_name : op } )

        flash(op_seq)

        # Operation pathways
        for op,(f0,f1) in operations.iteritems():

            # r0 (tens place), r1 (ones place)
            input_layers = { name : nvmnet.layers[name]
                for name in [self.x_name, self.y_name] }
            x_seq = Sequencer(nvmnet.layers[self.x_name], input_layers)
            y_seq = Sequencer(nvmnet.layers[self.y_name], input_layers)

            # encode arithmetic tables
            for a in range(10):
                for b in range(10):
                    input_states = {self.x_name : str(a), self.y_name : str(b)}
                    try : v0,v1 = f0(a,b), f1(a,b)
                    except ZeroDivisionError: v0,v1 = 'null','null'
                    x_seq.add_transit(str(v0), **input_states)
                    y_seq.add_transit(str(v1), **input_states)

            flash(x_seq, "_%s" % op)
            flash(y_seq, "_%s" % op)
