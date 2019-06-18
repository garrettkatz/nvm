import sys
sys.path.append('../nvm')

from random import randint, choice, sample, gauss
from math import sqrt, asin
from itertools import chain

from layer import Layer
from activator import *
from coder import Coder
from learning_rules import rehebbian

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from test_abduction import test_data as abduction_test_data
from test_abduction import build_fsm, abduce


class GraphNet:
    def __init__(self, N, pad, mask_frac, stabil=10):
        self.stabil = stabil
        self.mask_frac = mask_frac

        ### Create layers
        size = N**2
        self.act = tanh_activator(pad, size)
        #self.act_mask = gate_activator(pad, size)
        #self.act_mask = logistic_activator(pad, size)
        self.act_mask = heaviside_activator(size)

        self.reg_layer, self.mem_layer, self.ptr_layer = (
            Layer(k, (N,N), self.act, Coder(self.act)) for k in "rmp")

        # Gating masks
        self.masks = { }
        self.w_mask = np.zeros((size,size))

        # Weight matrices
        self.w_pp = np.zeros((size,size))
        self.w_mm = np.zeros((size,size))
        self.w_pm = np.zeros((size,size))
        self.w_mp = np.zeros((size,size))

        # Dummy bias to avoid extra memory allocation
        self.dummy_bias = np.zeros((size, 1))

        # Current keys/values stored
        self.keys = set()
        self.values = set()


    def gen_mask(self, size):
        m = np.zeros((size,1))
        m[np.random.choice(size, int(size / self.mask_frac), replace=False)] = 1.
        return m


    def learn_mappings(self, mappings):
        self.learn([(k,start,v) for start,m in mappings.items() for k,v in m])

    def learn(self, kfts):
        kfts = tuple(kfts)
        key_syms, from_syms, to_syms = zip(*kfts)
        mem_syms = list(set(from_syms + to_syms))

        # Construct masks for missing keys
        missing_keys = tuple(k for k in key_syms if k not in self.keys)
        self.masks.update({
            key_sym : self.gen_mask(self.reg_layer.size)
            for key_sym in missing_keys
        })
        self.keys.update(missing_keys)

        # Learn masks
        if len(missing_keys) > 0:
            X = self.reg_layer.encode_tokens(missing_keys)
            Y_mem = np.concatenate(
                tuple(self.masks[k] for k in missing_keys), axis=1)

            self.w_mask += rehebbian(self.w_mask, self.dummy_bias,
                X, Y_mem, self.act, self.act_mask)[0]

        # Replace masks with reconstructed versions
        self.masks = {
            key : self.act_mask.f(
                    self.w_mask.dot(
                        self.reg_layer.coder.encode(key)))
            for key in self.masks
        }

        # Learn recurrent weights
        X = self.ptr_layer.encode_tokens(mem_syms)
        self.w_pp += rehebbian(self.w_pp, self.dummy_bias, X, X, self.act, self.act)[0]

        X = self.mem_layer.encode_tokens(mem_syms)
        self.w_mm += rehebbian(self.w_mm, self.dummy_bias, X, X, self.act, self.act)[0]

        self.values.update(mem_syms)

        #  Learn inter-regional weights
        # mem -> ptr
        X = np.multiply(
            self.mem_layer.encode_tokens(from_syms),
            np.concatenate(tuple(self.masks[k] for k in key_syms), axis=1))
        Y = np.multiply(
            self.ptr_layer.encode_tokens(to_syms),
            np.concatenate(tuple(self.masks[k] for k in key_syms), axis=1))
        self.w_pm += rehebbian(self.w_pm, self.dummy_bias, X, Y, self.act, self.act)[0]

        # ptr -> mem
        X = self.ptr_layer.encode_tokens(to_syms)
        Y = self.mem_layer.encode_tokens(to_syms)
        self.w_mp += rehebbian(self.w_mp, self.dummy_bias, X, Y, self.act, self.act)[0]




    def test_recovery(self, mappings):
        mem_states = list(mappings.keys()) + list(
            set(v for m in mappings.values() for k,v in m))

        correct,total = 0,0

        for mask in self.masks.values():
            for tok in mem_states:
                complete = self.ptr_layer.coder.encode(tok)
                partial = np.multiply(complete, mask)
                y = self.act.f(self.w_pp.dot(partial))

                # Stabilize
                for _ in range(self.stabil):
                    old_y = y
                    y = self.act.f(self.w_pp.dot(y))
                    if np.array_equal(y, old_y):
                        break
                out = self.ptr_layer.coder.decode(y)

                # Check output
                correct += (out == tok)
                total += 1
        return float(correct) / total

    def test_traversal(self, mappings):
        reg_states = list(set(k for m in mappings.values() for k,v in m))
        key_pairs = { k:set() for k in reg_states }
        for start,m in mappings.items():
            for k,v in m:
                key_pairs[k].add((start,v))

        correct,total = 0,0

        stats = []

        for inp,s in key_pairs.items():
            # Masks have already been replaced with reconstructed versions
            # Retrieve to save computations
            mask = self.masks[inp]

            for start,end in s:
                start_pat = self.mem_layer.coder.encode(start)

                # Compute ptr activation
                x = np.multiply(start_pat, mask)
                y = np.multiply(self.act.f(self.w_pm.dot(x)), mask)

                # Stabilize ptr
                for _ in range(self.stabil):
                    old_y = y
                    y = self.act.f(self.w_pp.dot(y))
                    if np.array_equal(y, old_y):
                        break

                # Compute mem activation
                y = self.act.f(self.w_mp.dot(y))

                # Stabilize mem
                for _ in range(self.stabil):
                    old_y = y
                    y = self.act.f(self.w_mm.dot(y))
                    if np.array_equal(y, old_y):
                        break

                out = self.mem_layer.coder.decode(y)

                # Check output
                correct += (out == end)
                total += 1

                #if out != end:
                #    diff = sum(np.sign(y) != np.sign(self.mem_layer.coder.encode(end)))
                #    print("Incorrect:",start,inp,end,out,diff,float(diff)/y.size)

        return float(correct) / total

    def test(self, mappings):
        return {
            "trans_acc" : self.test_traversal(mappings),
            "recall_acc" : self.test_recovery(mappings) }

def print_results(prefix, results):
    print(
        ("%7s" % prefix) +
        " ".join("%12.4f" % results[k]
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

def test(N, pad, mask_frac, mappings):
    n = GraphNet(N, pad, mask_frac)
    n.learn_mappings(mappings)
    return n.test(mappings)

def print_header():
    print("       " + " ".join("%12s" % x for x in ["trans_acc", "recall_acc"]))

def test_random_networks(N, pad, mask_frac):
    print_header()

    num_nodes = N * 2
    for p in [0.1, 0.25, 0.5, 0.75, 1.0]:
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

        n = GraphNet(N, pad, mask_frac)
        n.learn_mappings(mappings)
        print_results("%d/%d" % (num_nodes, len(net.edges)), n.test(mappings))
    print("")

def test_machines(N, pad, mask_frac):
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
    print_results("Machine A", test(N, pad, mask_frac, mappings))

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

    print_results("Machine B", test(N, pad, mask_frac, mappings))
    print("")

    mappings = {
        "0" : [('A', "1"), ('S', "16"), ('T', "10"), ('B', "4"), ('Y', "7"), ('D', "9"), ('W', "12")],
        "3" : [],
        "10" : [('E', "11")],
        "4" : [('C', "5")],
        "11" : [],
        "5" : [],
        "12" : [('U', "13")],
        "1" : [('B', "2"), ('Z', "6")],
        "13" : [],
        "8" : [],
        "6" : [],
        "14" : [('C', "15")],
        "7" : [('C', "8")],
        "15" : [],
        "9" : [('E', "14")],
        "2" : [('C', "3")],
        "16" : [('A', "17")],
        "17" : [],
    }
    inputs = set()
    for st,trans in mappings.items():
        for inp,x in trans:
            inputs.add(inp)

    for st,trans in mappings.items():
        for inp in inputs:
            if inp not in trans:
                trans.append((inp, "NULL"))

    print("Causal Knowledge FSM:")
    print_results("", test(N, pad, mask_frac, mappings))
    print("")

def test_param_explore(N, pad, mask_frac):
    num_states = N * 2
    num_inputs = int(N ** 0.5)
    num_trans = int(N ** 0.5)

    print("num_states = %d" % num_states)
    print("num_inputs = %d" % num_inputs)
    print("num_trans = %d" % num_trans)
    print("")

    print_header()

    print("mask_frac")
    for x in [N/4, N/2, N, N*2]:
        x = int(x ** 0.5)
        mappings = gen_mappings(num_states, num_inputs, num_trans)
        print_results(x, test(N, pad, x, mappings))
    print("")

    print("num_states")
    for x in [N/2, N, N*2, N * 4]:
        x = int(x)
        mappings = gen_mappings(x, num_inputs, num_trans)
        print_results(x, test(N, pad, mask_frac, mappings))
    print("")

    print("num_inputs")
    for x in [N/2, N, N*2, N * 4, N * 8]:
        x = int(x)
        mappings = gen_mappings(num_states, x, num_trans)
        print_results(x, test(N, pad, mask_frac, mappings))
    print("")

    print("num_trans")
    for x in [N/8, N/4, N/2, N, N*2]:
        x = int(x)
        mappings = gen_mappings(max(num_states,x+1), max(num_inputs,x), x)
        print_results(x, test(N, pad, mask_frac, mappings))
    print("")

def test_traj(N, pad, mask_frac):
    # Create field
    field_states = [("f%d" % i) for i in range(N * 2)]

    # Create heads
    heads = [("h%d" % i) for i in range(N)]
    traj_length = N

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

    print_header()
    #for k,v in mappings.items(): print(k,v)
    print_results("Traj", test(N, pad, mask_frac, mappings))

    # Use numerical indices from head to each state
    mappings = {}
    for h,traj in zip(heads,trajs):
        if h not in mappings:
            mappings[h] = []
        mappings[h] = list(enumerate(traj))

    #for k,v in mappings.items(): print(k,v)
    print_results("Indexed", test(N, pad, mask_frac, mappings))

    # Use states as index into head
    mappings = {}
    for pre,h,post in pairs:
        if h not in mappings:
            mappings[h] = [ (pre, post) ]
        else:
            mappings[h].append((pre, post))

    #for k,v in mappings.items(): print(k,v)
    print_results("Rev-ndx", test(N, pad, mask_frac, mappings))
    print("")

def test_abduce(N, pad, mask_frac):
    for knowledge, seq, answer in abduction_test_data:
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
                'parent' : "NULL" if s.parent is None else str(s.parent),
                'transitions' : [ inp for inp in s.transitions ],
                'causes' : [ str(c) for c in s.causes ],
            } for s in fsm_states
        })
        for s in fsm_states:
            data[str(s)].update({
                inp : str(s2) for inp,s2 in s.transitions.items() })

        data.update({
            str(t) : {
                'previous' : "NULL" if t.previous is None else str(t.previous),
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
                str(c) : str(c) for s,c in c.cache.items() })

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
                            pairs.append((value[-1], parent, "NULL"))
                        else:
                            pairs.append((parent, key, "NULL"))
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
        print_results("", test(N, pad, mask_frac, mappings))
        print()

# Parameters
N = 32
pad = 0.0001

mask_frac = int(N ** 0.5)

print("N=%d" % N)
print("mask_frac = %d" % mask_frac)
print()

test_random_networks(N, pad, mask_frac)
test_machines(N, pad, mask_frac)
test_param_explore(N, pad, mask_frac)
test_traj(N, pad, mask_frac)
test_abduce(N, pad, mask_frac)
