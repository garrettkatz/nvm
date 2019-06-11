import sys
sys.path.append('../nvm')

from random import randint, choice, sample, gauss
from math import sqrt, asin

from layer import Layer
from activator import *
from coder import Coder
from learning_rules import rehebbian

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def gen_weight_mask(N, p):
    m = np.zeros((N,N))
    pos = [make_rand_vector(4) for _ in range(N)]

    for i, p1 in enumerate(pos):
        for j, p2 in enumerate(pos):
            if p1 != p2:
                d = sqrt(sum((x-y)**2 for (x,y) in zip(p1,p2)))
                m[i][j] = 2* asin(d/2)

    thresh = sorted(x for x in m.reshape(-1) if x > 0.0)[int((N**2-N)*p)]
    return m < thresh



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
        m[np.random.choice(size, size / self.mask_frac, replace=False)] = 1.
        return m


    def learn_mappings(self, mappings):
        self.learn([(k,start,v) for start,m in mappings.items() for k,v in m])

    def learn(self, kfts):
        kfts = tuple(kfts)
        key_syms, from_syms, to_syms = zip(*kfts)

        # Construct masks for missing keys
        missing_keys = tuple(k for k in key_syms if k not in self.keys)
        self.masks.update({
            key_sym : self.gen_mask(self.reg_layer.size)
            #key_sym : self.act_mask.f(self.reg_layer.coder.encode(key_sym))
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
        X = self.ptr_layer.encode_tokens(from_syms + to_syms)
        self.w_pp += rehebbian(self.w_pp, self.dummy_bias, X, X, self.act, self.act)[0]

        X = self.mem_layer.encode_tokens(from_syms + to_syms)
        self.w_mm += rehebbian(self.w_mm, self.dummy_bias, X, X, self.act, self.act)[0]

        self.values.update(from_syms + to_syms)

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
        mem_states = mappings.keys() + list(
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
        mappings[f] = zip(t,s)

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

def test_net(N, pad, mask_frac):
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




# Parameters
N = 16
pad = 0.0001

mask_frac = int(N ** 0.5)
num_states = N * 2
num_inputs = int(N ** 0.5)
num_trans = int(N ** 0.5)

print("N=%d" % N)
print("mask_frac = %d" % mask_frac)
print("num_states = %d" % num_states)
print("num_inputs = %d" % num_inputs)
print("num_trans = %d" % num_trans)
print("")
print("       " + " ".join("%12s" % x for x in ["trans_acc", "recall_acc"]))

#mappings = gen_mappings(num_states, num_inputs, num_trans)
#print_results("", test(N, pad, mask_frac, mappings))
test_net(N, pad, mask_frac)
print("")


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
print("Machine A")
print_results("", test(N, pad, mask_frac, mappings))

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

print("Machine B")
print_results("", test(N, pad, mask_frac, mappings))
print("")




print("mask_frac")
for x in [N/4, N/2, N, N*2]:
    x = int(x ** 0.5)
    mappings = gen_mappings(num_states, num_inputs, num_trans)
    print_results(x, test(N, pad, x, mappings))
print("")

print("num_states")
for x in [N/2, N, N*2, N * 4]:
    mappings = gen_mappings(x, num_inputs, num_trans)
    print_results(x, test(N, pad, mask_frac, mappings))
print("")

print("num_inputs")
for x in [N/2, N, N*2, N * 4, N * 8]:
    mappings = gen_mappings(num_states, x, num_trans)
    print_results(x, test(N, pad, mask_frac, mappings))
print("")

print("num_trans")
for x in [N/8, N/4, N/2, N, N*2]:
    mappings = gen_mappings(max(num_states,x+1), max(num_inputs,x), x)
    print_results(x, test(N, pad, mask_frac, mappings))
print("")
