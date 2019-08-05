import sys
sys.path.append('../nvm')

from random import randint, choice, sample, gauss
from math import sqrt, asin
from itertools import chain

from activator import tanh_activator, heaviside_activator
from coder import Coder

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#from test_abduction import test_data as abduction_test_data
#from test_abduction import build_fsm, abduce

################################################################################

from syngen import Network, Environment, create_io_callback, FloatArray
from syngen import get_cpu, get_gpus, interrupt_engine

from syngen_nvm import make_custom_input_module, make_custom_output_module

def build_layer(layer_name, model, rows, cols, pad):
    return {
        "name" : layer_name,
        "neural model" : model,
        "rows" : rows,
        "columns" : cols,
        "pad" : pad
    }
def build_gate(to_name, from_name):
    return {
        "name" : "%s<%s_%s" % (to_name, from_name, "gate"),
        "from layer" : "g",          #TBD
        "to layer" : to_name,
        "type" : "subset",
        "subset config" : {
            "from row end" : 1,
            "from column start" : 0, #TBD
            "from column end" : 0,   #TBD
            "to row end" : 1,
            "to column end" : 1,
        },
        "plastic" : False,
        "gate" : True
    }
def build_learning_gate(to_name, from_name):
    return {
        "name" : "%s<%s_%s" % (to_name, from_name, "learning"),
        "from layer" : "g",          #TBD
        "to layer" : to_name,
        "type" : "subset",
        "subset config" : {
            "from row end" : 1,
            "from column start" : 0, #TBD
            "from column end" : 0,   #TBD
            "to row end" : 1,
            "to column end" : 1,
        },
        "plastic" : False,
        "learning" : True
    }
def build_sparse(to_name, from_name, fraction, pad):
    return {
        "name" : "%s<%s_%s" % (to_name, from_name, "weights"),
        "from layer" : from_name,
        "to layer" : to_name,
        "type" : "fully connected",
        "sparse" : True,
        "opcode" : "add",
        "weight config" : {
            "type" : "flat",
            "weight" : 0.001,
            "fraction" : fraction
        },
        "gated" : True,
        "plastic" : True,
        "norm" : (1. - pad) ** 2
    }
def build_conv(to_name, from_name, field_dim, pad):
    return {
        "name" : "%s<%s_%s" % (to_name, from_name, "weights"),
        "from layer" : from_name,
        "to layer" : to_name,
        "type" : "convergent",
        "arborized config" : {
            "field size" : field_dim,
            "stride" : 1,
            "wrap" : True,
        },
        "opcode" : "add",
        "weight config" : {
            "type" : "flat",
            "weight" : 0.
        },
        "gated" : True,
        "plastic" : True,
        "norm" : (1. - pad) ** 2
    }
def build_fc(to_name, from_name, N, pad):
    return {
        "name" : "%s<%s_%s" % (to_name, from_name, "weights"),
        "from layer" : from_name,
        "to layer" : to_name,
        "type" : "fully connected",
        "opcode" : "add",
        "weight config" : {
            "type" : "flat",
            "weight" : 0.
        },
        "gated" : True,
        "plastic" : True,
        "norm" : (1. - pad) ** 2
    }
def build_decay(to_name):
    return {
        "name" : "%s<%s_%s" % (to_name, to_name, "decay"),
        "from layer" : "g",          #TBD
        "to layer" : to_name,
        "type" : "subset",
        "subset config" : {
            "from row end" : 1,
            "from column start" : 0, #TBD
            "from column end" : 0,   #TBD
            "to row end" : 1,
            "to column end" : 1,
        },
        "plastic" : False,
        "decay" : True
    }
def build_gain(to_name):
    return {
        "name" : "%s<%s_%s" % (to_name, to_name, "gain"),
        "from layer" : to_name,
        "to layer" : to_name,
        "type" : "one to one",
        "opcode" : "add",
        "weight config" : {
            "type" : "flat",
            "weight" : 4.952213997042802
        },
        "plastic" : False,
        "gated" : True
    }
def build_oto(to_name, from_name):
    return {
        "name" : "%s<%s_%s" % (to_name, from_name, "weights"),
        "from layer" : from_name,
        "to layer" : to_name,
        "type" : "one to one",
        "opcode" : "mult",
        "weight config" : {
            "type" : "flat",
            "weight" : 1
        },
        "gated" : True,
        "plastic" : False,
    }

def build_unit(prefix, N, structure_name, conv=1., pad=0.0001):
    conv = max(0., min(conv, 1.))

    ### LAYERS ###
    layer_configs = []

    m = prefix+"m"
    p = prefix+"p"
    c = prefix+"c"

    # Memory/pointer/context layers
    layer_configs.append(build_layer(m, "nvm", N, N, pad))
    layer_configs.append(build_layer(p, "nvm", N, N, pad))
    layer_configs.append(build_layer(c, "nvm_heaviside", N, N, pad))

    ### CONNECTIONS ###
    connections = []

    # Autoassociative and heteroassociative full weight matrices
    for (to_name, from_name) in [(m,m), (p,m), (m,p)]:
        connections.append(build_gate(to_name, from_name))
        connections.append(build_learning_gate(to_name, from_name))

        # Use local connectivity for autoassociative matrix
        if conv == 1.:
            connections.append(build_fc(to_name, from_name, N, pad))
        else:
            #connections.append(build_conv(to_name, from_name, conv * conv * N, pad))
            connections.append(build_sparse(to_name, from_name, conv ** 4, pad))

    # Gain connections
    for to_name in [c, m, p]:
        connections.append(build_decay(to_name))
        connections.append(build_gain(to_name))

    # Context gating
    for (to_name, from_name) in [(m,c), (p,c)]:
        connections.append(build_gate(to_name, from_name))
        connections.append(build_oto(to_name, from_name))

    # Set structures
    for conn in connections:
        conn["from structure"] = structure_name
        conn["to structure"] = structure_name

    return layer_configs, connections


class GraphNet:
    def __init__(self, N, mask_frac, conv=1., stabil=10):
        N = int(N / conv)

        self.stabil = stabil
        self.mask_frac = mask_frac

        self.size = N**2
        self.mask_size = int(self.size / self.mask_frac)
        pad = 0.0001

        # Build mem/ptr/ctx unit
        self.prefix = "test"
        layer_configs,connections = build_unit(self.prefix, N, "graph_net", conv, pad)

        # Assign gate indices
        self.gate_layer_name = "g"
        self.gates = {}
        for conn in connections:
            if any(conn.get(key, False) for key in ["gate", "decay", "learning"]):
                conn["from layer"] = self.gate_layer_name
                conn["subset config"]["from column start"] = len(self.gates)
                conn["subset config"]["from column end"] = len(self.gates)+1
                self.gates[conn["name"]] = len(self.gates)

        # Build gate layer
        layer_configs.append(build_layer(
            self.gate_layer_name, "nvm_heaviside", 1, len(self.gates), pad))

        structure = {"name" : "graph_net",
                     "type" : "parallel",
                     "layers": layer_configs}

        self.net = Network({
            "structures" : [structure],
            "connections" : connections})

        ### Create activators and coders
        self.act = tanh_activator(pad, self.size)
        self.act_h = heaviside_activator(self.size)

        self.layer_names = [
            self.prefix+"m",
            self.prefix+"p",
            self.prefix+"c",
            self.gate_layer_name
        ]

        self.acts = {
            self.prefix+"m" : self.act,
            self.prefix+"p" : self.act,
            self.prefix+"c" : self.act_h,
            self.gate_layer_name : self.act_h,
        }
        self.coders = {
            self.prefix+"m" : Coder(self.act),
            self.prefix+"p" : Coder(self.act),
            self.prefix+"c" : Coder(self.act_h),
            self.gate_layer_name : Coder(self.act_h),
        }

    def add_gate_pattern(self, name, on):
        arr = np.zeros(len(self.gates))
        for to_prefix,to_name,from_prefix,from_name,typ in on:
            i = "%s%s<%s%s_%s" % (to_prefix,to_name,from_prefix,from_name,typ)
            arr[self.gates[i]] = 1.
        self.coders[self.gate_layer_name].encode(name, arr)

    def execute(self, visualizer=False, input_iters={}, output_iters={}, args={}):
        test_state = {
            k :  {
                # Accuracy of Hebbian mem recovery
                "w_correct" : 0.,
                # Symbolic final correct
                "correct" : 0,
                # Total tests
                "total" : 0,
            } for k,v in output_iters.items()
        }

        def input_callback(layer_name, data):
            try:
                inp = next(input_iters[layer_name])
                if inp is not None:
                    np.copyto(data,
                        self.acts[layer_name].g(
                            self.coders[layer_name].encode(inp)).flat)
            except StopIteration:
                interrupt_engine()
            except Exception as e:
                print(e)
                interrupt_engine()

        # Collects and validates memory state
        def output_callback(layer_name, data):
            try:
                tok = next(output_iters[layer_name])
                if tok is not None:
                    out = self.coders[layer_name].decode(data)

                    # Check output
                    if out == tok:
                        test_state[layer_name]["w_correct"] += 1
                        test_state[layer_name]["correct"] += 1
                    else:
                        test_state[layer_name]["w_correct"] += (
                            np.sum(np.sign(data) == np.sign(
                                self.acts[layer_name].g(
                                    self.coders[layer_name].encode(tok))).flat)
                            / data.size)
                    test_state[layer_name]["total"] += 1
            except StopIteration:
                interrupt_engine()
            except Exception as e:
                print(e)
                interrupt_engine()

        modules = [
            make_custom_input_module(
                "graph_net", list(input_iters.keys()), "icb", input_callback),
            make_custom_output_module(
                "graph_net", list(output_iters.keys()), "ocb", output_callback)
        ]

        if visualizer:
            modules.append( {
                "type" : "visualizer",
                "negative" : True,
                "layers" : [
                    {"structure": "graph_net", "layer": layer_name}
                        for layer_name in self.layer_names] })

        default_args = {
            "multithreaded" : True,
            "worker threads" : 0,
            #"devices" : get_cpu(),
            "iterations" : 0,
            "refresh rate" : 0,
            #"verbose" : True,
            "learning flag" : False}

        for key,val in default_args.items():
            if key not in args:
                args[key] = val

        # Clear prior activity
        for layer_name in self.layer_names:
            self.net.get_neuron_data(
                "graph_net", layer_name, "output").to_np_array().fill(0.)

        env = Environment({"modules" : modules})
        report = self.net.run(env, args)
        env.free()
        return report, test_state

    def learn(self, mappings):
        kfts = [(k,start,v) for start,m in mappings.items() for k,v in m]

        # Construct masks
        for k,start,v in kfts:
            mask = np.zeros((self.size,1))
            mask[np.random.choice(
                self.size, self.mask_size, replace=False)] = 1.
            self.coders[self.prefix+"c"].encode(k, mask)

        # Set up I/O sequence
        mems,ptrs,ctxs,gates = [],[],[],[]

        self.add_gate_pattern("learn1", [])
        self.add_gate_pattern("learn2", 
            [(self.prefix, "p", self.prefix, "m", "learning"),
             (self.prefix, "m", self.prefix, "m", "decay")])
        self.add_gate_pattern("learn3", [])
        self.add_gate_pattern("learn4", 
            [(self.prefix, "m", self.prefix, "m", "learning"),
             (self.prefix, "m", self.prefix, "c", "gate"),
             (self.prefix, "p", self.prefix, "c", "gate")])
        self.add_gate_pattern("learn5", 
            [(self.prefix, "m", self.prefix, "p", "learning"),
             (self.prefix, "c", self.prefix, "c", "decay"),
             (self.prefix, "m", self.prefix, "m", "decay"),
             (self.prefix, "p", self.prefix, "p", "decay")])
        
        # Learn from/to ptr/mem transitions
        for key_sym,from_sym,to_sym in kfts:
            # Load mem/ptr/ctx
            mems.append(from_sym)
            ptrs.append(from_sym)
            ctxs.append(key_sym)
            gates.append("learn1")

            # Learn mem->ptr, close mem gain
            # learn_pm, mem_decay
            mems.append(None)
            ptrs.append(None)
            ctxs.append(None)
            gates.append("learn2")

            # Load mem
            mems.append(to_sym)
            ptrs.append(None)
            ctxs.append(None)
            gates.append("learn3")

            # Learn mem->mem, open ctx->mem and ctx->ptr
            # learn_mm, mem_ctx, ptr_ctx
            mems.append(None)
            ptrs.append(None)
            ctxs.append(None)
            gates.append("learn4")

            # Learn ptr->mem, close all gain
            # learn_mp, ctx_decay, mem_decay, ptr_decay
            mems.append(None)
            ptrs.append(None)
            ctxs.append(None)
            gates.append("learn5")

        input_iters = {
            self.prefix+"m" : iter(mems),
            self.prefix+"p" : iter(ptrs),
            self.prefix+"c" : iter(ctxs),
            self.gate_layer_name : iter(gates)
        }

        self.execute(input_iters=input_iters)

    def test_recovery(self, mappings):
        # Retrieve all memory states
        mem_states = set(k for k in mappings.keys()).union(
            set(v for m in mappings.values() for k,v in m))

        # Set up I/O sequence
        mems,ctxs,gates = [],[],[]
        outputs = []

        self.add_gate_pattern("recov1", 
            [(self.prefix, "c", self.prefix, "c", "decay"),
             (self.prefix, "p", self.prefix, "p", "decay"),
             (self.prefix, "m", self.prefix, "c", "gate")])
        self.add_gate_pattern("recov2", 
            [(self.prefix, "m", self.prefix, "m", "gate"),
             (self.prefix, "c", self.prefix, "c", "decay"),
             (self.prefix, "m", self.prefix, "m", "decay"),
             (self.prefix, "p", self.prefix, "p", "decay")])
        self.add_gate_pattern("recov3", 
            [(self.prefix, "c", self.prefix, "c", "decay"),
             (self.prefix, "m", self.prefix, "m", "decay"),
             (self.prefix, "p", self.prefix, "p", "decay")])

        for tok in mem_states:
            for key_sym in self.coders[self.prefix+"c"].list_tokens():
                # Load input, open mem->mem, close all gain
                # mem_mem, ctx_decay, mem_decay, ptr_decay
                mems.append(tok)
                ctxs.append(key_sym)
                gates.append("recov1")
                outputs.append(None)

                # Stabilize
                for _ in range(self.stabil):
                    mems.append(None)
                    ctxs.append(None)
                    gates.append("recov2")
                    outputs.append(None)

                # Read output, close all gain
                # ctx_decay, mem_decay, ptr_decay
                mems.append(None)
                ctxs.append(None)
                gates.append("recov3")
                outputs.append(tok)

        input_iters = {
            self.prefix+"m" : iter(mems),
            self.prefix+"c" : iter(ctxs),
            self.gate_layer_name : iter(gates)
        }
        output_iters = {
            self.prefix+"m" : iter(outputs),
        }

        report,test_state = self.execute(
            input_iters=input_iters, output_iters=output_iters)

        correct = test_state[self.prefix+"m"]["correct"]
        w_correct = test_state[self.prefix+"m"]["w_correct"]
        total = test_state[self.prefix+"m"]["total"]

        return (float(correct) / total, w_correct / total)

    def test_transit(self, mappings):
        # Set up I/O sequence
        mems,ctxs,gates = [],[],[]
        outputs = []

        self.add_gate_pattern("trans1", 
            [(self.prefix, "p", self.prefix, "p", "decay")])
        self.add_gate_pattern("trans2", 
            [(self.prefix, "p", self.prefix, "m", "gate"),
             (self.prefix, "m", self.prefix, "m", "decay"),
             (self.prefix, "p", self.prefix, "p", "decay"),
             (self.prefix, "p", self.prefix, "c", "gate")])
        self.add_gate_pattern("trans3", 
            [(self.prefix, "m", self.prefix, "p", "gate"),
             (self.prefix, "m", self.prefix, "m", "decay"),
             (self.prefix, "p", self.prefix, "p", "decay"),
             (self.prefix, "m", self.prefix, "c", "gate")])
        self.add_gate_pattern("trans4", 
            [(self.prefix, "m", self.prefix, "m", "gate"),
             (self.prefix, "c", self.prefix, "c", "decay"),
             (self.prefix, "m", self.prefix, "m", "decay"),
             (self.prefix, "p", self.prefix, "p", "decay")])
        self.add_gate_pattern("trans5", 
            [(self.prefix, "c", self.prefix, "c", "decay"),
             (self.prefix, "m", self.prefix, "m", "decay"),
             (self.prefix, "p", self.prefix, "p", "decay")])

        for start,m in mappings.items():
            for inp,end in m:
                # Decay mem/ptr
                mems.append(start)
                ctxs.append(inp)
                gates.append("trans1")
                outputs.append(None)

                # p<m, decay mem/ptr, p<c
                mems.append(None)
                ctxs.append(None)
                gates.append("trans2")
                outputs.append(None)

                # m<p, decay mem/ptr, m<c
                mems.append(None)
                ctxs.append(None)
                gates.append("trans3")
                outputs.append(None)

                for _ in range(self.stabil):
                    # m<m
                    mems.append(None)
                    ctxs.append(None)
                    gates.append("trans4")
                    outputs.append(None)

                mems.append(None)
                ctxs.append(None)
                # Clear
                gates.append("trans5")
                outputs.append(end)

        input_iters = {
            self.prefix+"m" : iter(mems),
            self.prefix+"c" : iter(ctxs),
            self.gate_layer_name : iter(gates)
        }
        output_iters = {
            self.prefix+"m" : iter(outputs)
        }

        report,test_state = self.execute(
            #visualizer=True, args={"refresh rate" : 10, "verbose" : True},
            input_iters=input_iters, output_iters=output_iters)

        correct = test_state[self.prefix+"m"]["correct"]
        w_correct = test_state[self.prefix+"m"]["w_correct"]
        total = test_state[self.prefix+"m"]["total"]

        return (float(correct) / total,
            w_correct / total)

    def test_traversal(self, trajs):
        # Set up I/O sequence
        mems,ctxs,gates = [],[],[]
        outputs = []

        self.add_gate_pattern("trav_clear", 
            [(self.prefix, "m", self.prefix, "m", "decay"),
             (self.prefix, "p", self.prefix, "p", "decay"),
             (self.prefix, "c", self.prefix, "c", "decay")])
        self.add_gate_pattern("trav_load", 
            [(self.prefix, "p", self.prefix, "p", "decay"),
             (self.prefix, "c", self.prefix, "c", "decay")])

        self.add_gate_pattern("trav1", 
            [(self.prefix, "p", self.prefix, "p", "decay")])
        self.add_gate_pattern("trav2", 
            [(self.prefix, "p", self.prefix, "m", "gate"),
             (self.prefix, "m", self.prefix, "m", "decay"),
             (self.prefix, "p", self.prefix, "p", "decay"),
             (self.prefix, "p", self.prefix, "c", "gate")])
        self.add_gate_pattern("trav3", 
            [(self.prefix, "m", self.prefix, "p", "gate"),
             (self.prefix, "m", self.prefix, "m", "decay"),
             (self.prefix, "p", self.prefix, "p", "decay"),
             (self.prefix, "m", self.prefix, "c", "gate")])
        self.add_gate_pattern("trav4", 
            [(self.prefix, "m", self.prefix, "m", "gate"),
             (self.prefix, "c", self.prefix, "c", "decay"),
             (self.prefix, "m", self.prefix, "m", "decay"),
             (self.prefix, "p", self.prefix, "p", "decay")])
        self.add_gate_pattern("trav5", 
            [(self.prefix, "c", self.prefix, "c", "decay"),
             (self.prefix, "p", self.prefix, "p", "decay")])

        for start,ctx_keys,traj in trajs:
            mems.append(None)
            ctxs.append(None)
            gates.append("trav_clear")
            outputs.append(None)

            mems.append(start)
            ctxs.append(None)
            gates.append("trav_load")
            outputs.append(None)

            for key,end in zip(ctx_keys,traj):
                # Decay mem/ptr
                mems.append(None)
                ctxs.append(key)
                gates.append("trav1")
                outputs.append(None)

                # p<m, decay mem/ptr, p<c
                mems.append(None)
                ctxs.append(None)
                gates.append("trav2")
                outputs.append(None)

                # m<p, decay mem/ptr, m<c
                mems.append(None)
                ctxs.append(None)
                gates.append("trav3")
                outputs.append(None)

                for _ in range(self.stabil):
                    # m<m
                    mems.append(None)
                    ctxs.append(None)
                    gates.append("trav4")
                    outputs.append(None)

                mems.append(None)
                ctxs.append(None)
                # Clear
                gates.append("trav5")
                outputs.append(end)

        input_iters = {
            self.prefix+"m" : iter(mems),
            self.prefix+"c" : iter(ctxs),
            self.gate_layer_name : iter(gates)
        }
        output_iters = {
            self.prefix+"m" : iter(outputs)
        }

        report,test_state = self.execute(
            #visualizer=True, args={"refresh rate" : 10, "verbose" : True},
            input_iters=input_iters, output_iters=output_iters)

        correct = test_state[self.prefix+"m"]["correct"]
        w_correct = test_state[self.prefix+"m"]["w_correct"]
        total = test_state[self.prefix+"m"]["total"]

        return (float(correct) / total,
            w_correct / total)

    def test(self, mappings):
        return {
            "trans_acc" : self.test_transit(mappings),
            "recall_acc" : self.test_recovery(mappings) }



def print_results(prefix, results):
    print(
        ("%7s" % prefix) +
        " ".join("      " + " / ".join("%6.4f" % r for r in results[k])
            for k in [
                "trans_acc", "recall_acc" ]))



def gen_mappings(num_states, num_inputs, num_trans):
    # Create finite state machine with input conditional transitions
    mem_states = [str(x) for x in range(num_states)]
    input_tokens = [str(x) for x in range(num_inputs)]

    # Encode transitions
    mappings = dict()
    for f in mem_states:
        others = [x for x in mem_states if x != f]
        s = np.random.choice(others, num_trans, replace=False)
        t = np.random.choice(input_tokens, num_trans, replace=False)
        mappings[f] = list(zip(t,s))

    return mappings

def table_to_mappings(table):
    return {
        str(i): [(table[i][j], str(j))
                    for j in range(len(table))
                        if table[i][j] is not None]
        for i in range(len(table))
    }

def test(N, mask_frac, mappings):
    n = GraphNet(N, mask_frac)
    n.learn(mappings)
    return n.test(mappings)

def print_header():
    print("       " + " ".join(
        "%21s" % x for x in [
        "trans_acc", "recall_acc"]))

def test_random_networks(N, mask_frac):
    print_header()

    num_nodes = N * 2
    for p in [0.1, 0.5]:
        net = nx.fast_gnp_random_graph(num_nodes, p, directed=True)
        #print(len(net.nodes), len(net.edges))

        edges = {}
        for u,v in net.edges:
            if u not in edges: edges[u] = []
            edges[u].append(v)

        mappings = {}
        keys = range(max(len(vs) for vs in edges.values()))
        for u,vs in edges.items():
            if u not in mappings: mappings[u] = [(i,v)
                for i,v in zip(sample(keys,len(vs)), vs)]
                #for i,v in enumerate(vs)]

        n = GraphNet(N, mask_frac)
        n.learn(mappings)
        print_results("%d/%d" % (num_nodes, len(net.edges)), n.test(mappings))
    print("")

def test_machines(N, mask_frac):
    print_header()

    table = [
        [None, "A", "B", "C", "D", "E", "F", "G"],
        ["D", None, "A", None, None, None, None, None],
        ["D", None, None, "B", None, "C", None, None],
        ["D", None, None, None, None, None, None, None],
        ["D", None, "C", None, "A", None, None, "B"],
        ["D", None, None, None, None, None, None, None],
        ["D", None, "A", "B", "C", None, None, None],
        ["D", None, None, None, "B", None, "A", None],
    ]

    mappings = table_to_mappings(table)
    print_results("Machine A", test(N, mask_frac, mappings))

    mappings = {
        "1" : [("A", "2"), ("B", "3"), ("C", "4"), ("D", "6")],
        "2" : [("A", "1")],
        "3" : [("B", "1")],
        "4" : [("A", "5")],
        "5" : [("C", "1")],
        "6" : [("A", "7"), ("B", "8"), ("C", "9")],
        "7" : [("B", "1")],
        "8" : [("A", "2")],
        "9" : [("A", "3")],
    }

    print_results("Machine B", test(N, mask_frac, mappings))
    print("")

def test_param_explore(N, mask_frac):
    num_states = N * 2
    num_inputs = int(N ** 0.5)
    num_trans = int(N ** 0.5)

    print("num_states = %d" % num_states)
    print("num_inputs = %d" % num_inputs)
    print("num_trans = %d" % num_trans)
    print("")

    print_header()

    print("mask_frac")
    for x in [mask_frac, mask_frac ** 2]:
        mappings = gen_mappings(num_states, num_inputs, num_trans)
        print_results(x, test(N, x, mappings))
    print("")

    print("num_states")
    for x in [N*2, N * 4]:
        x = int(x)
        mappings = gen_mappings(x, num_inputs, num_trans)
        print_results(x, test(N, mask_frac, mappings))
    print("")

    print("num_inputs")
    for x in [N * 4, N * 8]:
        x = int(x)
        mappings = gen_mappings(num_states, x, num_trans)
        print_results(x, test(N, mask_frac, mappings))
    print("")

    print("num_trans")
    for x in [N, N*2]:
        x = int(x)
        mappings = gen_mappings(max(num_states,x+1), max(num_inputs,x), x)
        print_results(x, test(N, mask_frac, mappings))
    print("")

def test_traj(N, mask_frac):
    # Create field
    field_states = [("f%d" % i) for i in range(N*2)]

    # Create heads
    heads = [("h%d" % i) for i in range(N*4)]
    traj_length = int(N / 2)

    print("field_states = %d" % len(field_states))
    print("heads        = %d" % len(heads))
    print("traj_length  = %d" % traj_length)
    print("")

    trajs = []
    pairs = []
    mappings = {}

    # Chain states together using head as key
    for h in heads:
        traj = np.random.choice(field_states, traj_length, replace=False)
        trajs.append(traj)

        for i in range(traj_length-1):
            pairs.append((traj[i], h, traj[i+1]))
        pairs.append((traj[-1], h, "NULL"))

    for pre,h,post in pairs:
        if pre not in mappings:
            mappings[pre] = [ (h, post) ]
        else:
            mappings[pre].append((h, post))

    n = GraphNet(N, mask_frac)
    n.learn(mappings)
    correct,w_correct = n.test_traversal(
        [(traj[0], [h for t in traj], list(traj[1:]) + ["NULL"])
            for h,traj in zip(heads,trajs)])
    print("Sequential traversals: %6.4f / %6.4f\n" % (correct, w_correct))
    return

    print_header()
    #for k,v in mappings.items(): print(k,v)
    print_results("Traj", test(N, mask_frac, mappings))

    # Use numerical indices from head to each state
    mappings = {}
    for h,traj in zip(heads,trajs):
        if h not in mappings:
            mappings[h] = []
        mappings[h] = list(enumerate(traj))

    #for k,v in mappings.items(): print(k,v)
    print_results("Indexed", test(N, mask_frac, mappings))

    # Use states as index into head
    mappings = {}
    for pre,h,post in pairs:
        if h not in mappings:
            mappings[h] = [ (pre, post) ]
        else:
            mappings[h].append((pre, post))

    #for k,v in mappings.items(): print(k,v)
    print_results("Rev-ndx", test(N, mask_frac, mappings))
    print("")

def test_abduce():
    test_data = []
    actions = 'ABC'
    causes = 'XYZSTUGHIJ'
    symbols = actions + causes

    p = [1. / len(actions) for a in actions]
    p = [x /sum(p) for x in p]

    knowledge = []
    for i in range(12):
        knowledge.append((
            sample(causes,1)[0] , tuple(
                symbols[i] for i in np.random.choice(len(p),2, p=p))))

    p = [.5 / len(actions) for a in actions] + [.5 / len(causes) for c in causes]
    p = [x /sum(p) for x in p]
    for i in range(8):
        knowledge.append((
            sample(causes,1)[0] , tuple(
                symbols[i] for i in np.random.choice(len(p),2, p=p))))

    print("Knowledge:")
    for k in knowledge:
        print("  " + str(k))

    seq = "".join([choice('ABC') for _ in range(256)])
    for l in (2, 4, 8, 16, 32):
        test_data.append(
            (
                knowledge,
                seq[:l],
                None
            ))

    for knowledge, seq, answer in test_data:
    #for knowledge, seq, answer in abduction_test_data:
        fsm = build_fsm(knowledge)
        timepoints, best_path = abduce(fsm, seq)
        fsm_states = list(iter(fsm))
        causes = [c for t in timepoints for c in t.causes]

        data = {
            #"curr_t" : str(timepoints[0]),
            #"fsm" : str(fsm),
        }

        data.update({
            str(s) : {
                'parent' : str(s.parent),
                'transitions' : [ inp for inp in s.transitions ],
                'causes' : [ str(c) for c in s.causes ],
            } for s in fsm_states
        })
        for s in fsm_states:
            data[str(s)].update({
                inp : str(s2) for inp,s2 in s.transitions.items() })

        data.update({
            str(t) : {
                'previous' : str(t.previous),
                'causes' : [ str(c) for c in t.causes ],
            } for t in timepoints
        })

        data.update({
            str(c) : {
                'identity' : c.identity,
                'start_t' : str(c.start_t),
                'end_t' : str(c.end_t),
                'source_state' : str(c.source_state),
                'cache' : [ str(s) for s in c.cache ],
                'effects' : [ str(e) for e in c.effects ],
            } for c in causes
        })
        for c in causes:
            data[str(c)].update({
                str(s) : str(c) for s,c in c.cache.items() })

        #for x in [str(x) for x in fsm_states + timepoints + causes]:
        #    print(x, data[x])

        pairs = []

        for parent,d in data.items():
            if type(d) is dict:
                for key,value in d.items():
                    if type(value) is list:
                        if len(value) > 0:
                            # Chain data together
                            pairs.append((parent, key, value[0]))
                            for (v1,v2) in zip(value, value[1:]):
                                pairs.append((v1,parent,v2))
                            pairs.append((value[-1], parent, "None"))
                        else:
                            pairs.append((parent, key, "None"))
                    else:
                        pairs.append((parent, key, value))

        mappings = {}
        for pre,key,post in pairs:
            if pre not in mappings:
                mappings[pre] = [ (key, post) ]
            else:
                mappings[pre].append((key, post))
        print(seq)
        #for k in sorted(mappings.keys()):
        #    print(k, mappings[k])

        mem_states = set()
        reg_states = set()
        for k,v in mappings.items():
            mem_states.add(k)
            for inp,res in v:
                reg_states.add(inp)
                mem_states.add(res)
        reg_states = reg_states - set(mem_states)
        print("Timepoints: ", len(timepoints))
        print("FSM states: ", len(fsm_states))
        print("Causes: ", len(causes))
        print("Memory states: ", len(mem_states))
        print("Register states: ", len(reg_states))
        print("Total states: ", len(mem_states.union(reg_states)))
        print("Total transitions: ", len(pairs))

        print()
        print_header()
        for N in [16, 24, 32]:
            print_results(N, test(N, (N**0.5), mappings))
        print()

# Parameters
for N in [16, 24, 32]:
    mask_frac = (N ** 0.5)

    print("-" * 80)
    print("N=%d" % N)
    print("mask_frac = %s" % mask_frac)
    print()

    test_random_networks(N, mask_frac)
    test_machines(N, mask_frac)
    test_param_explore(N, mask_frac)
    test_traj(N, mask_frac)
print("-" * 80)
#test_abduce()
