import sys
sys.path.append('../nvm')

from random import randint

from layer import Layer
from activator import *
from coder import Coder
from learning_rules import rehebbian

import numpy as np

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
        self.mem_masks = { }
        self.ptr_masks = { }

        self.w_mem_mask = np.zeros((size,size))
        self.w_ptr_mask = np.zeros((size,size))

        # Weight matrices
        self.w_r = np.zeros((size,size))
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
        for key_sym in missing_keys:
            self.mem_masks[key_sym] = self.gen_mask(self.reg_layer.size)
            self.ptr_masks[key_sym] = self.gen_mask(self.reg_layer.size)
        self.keys.update(missing_keys)

        # Learn masks
        if len(missing_keys) > 0:
            X = self.reg_layer.encode_tokens(missing_keys)
            Y_mem = np.concatenate(
                tuple(self.mem_masks[k] for k in missing_keys), axis=1)
            Y_ptr = np.concatenate(
                tuple(self.ptr_masks[k] for k in missing_keys), axis=1)

            self.w_mem_mask += rehebbian(self.w_mem_mask, self.dummy_bias,
                X, Y_mem, self.act, self.act_mask)[0]
            self.w_ptr_mask += rehebbian(self.w_ptr_mask, self.dummy_bias,
                X, Y_ptr, self.act, self.act_mask)[0]

        # Learn recurrent weights
        X = self.ptr_layer.encode_tokens(from_syms + to_syms)
        self.w_r += rehebbian(self.w_r, self.dummy_bias, X, X, self.act, self.act)[0]
        self.values.update(from_syms + to_syms)

        #  Learn inter-regional weights
        # mem -> ptr
        X = np.multiply(
            self.mem_layer.encode_tokens(from_syms),
            np.concatenate(tuple(self.mem_masks[k] for k in key_syms), axis=1))
        Y = np.multiply(
            self.ptr_layer.encode_tokens(to_syms),
            np.concatenate(tuple(self.ptr_masks[k] for k in key_syms), axis=1))
        self.w_pm += rehebbian(self.w_pm, self.dummy_bias, X, Y, self.act, self.act)[0]

        # ptr -> mem
        X = self.ptr_layer.encode_tokens(to_syms)
        Y = self.mem_layer.encode_tokens(to_syms)
        self.w_mp += rehebbian(self.w_mp, self.dummy_bias, X, Y, self.act, self.act)[0]




    def test_recovery(self, mappings):
        mem_states = mappings.keys() + list(
            set(v for m in mappings.values() for k,v in m))

        correct,total = 0,0

        for mask in self.ptr_masks.values():
            for tok in mem_states:
                complete = self.ptr_layer.coder.encode(tok)
                partial = np.multiply(complete, mask)
                y = self.act.f(self.w_r.dot(partial))

                # Stabilize
                for _ in range(self.stabil):
                    old_y = y
                    y = self.act.f(self.w_r.dot(y))
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

        for inp,s in key_pairs.items():
            inp_pat = self.reg_layer.coder.encode(inp)
            mem_mask = self.act_mask.f(self.w_mem_mask.dot(inp_pat))
            ptr_mask = self.act_mask.f(self.w_ptr_mask.dot(inp_pat))

            for start,end in s:
                start_pat = self.mem_layer.coder.encode(start)

                # Compute ptr activation
                x = np.multiply(start_pat, mem_mask)
                y = np.multiply(self.act.f(self.w_pm.dot(x)), ptr_mask)

                # Stabilize ptr
                for _ in range(self.stabil):
                    old_y = y
                    y = self.act.f(self.w_r.dot(y))
                    if np.array_equal(y, old_y):
                        break

                # Compute mem activation
                y = self.act.f(self.w_mp.dot(y))
                out = self.mem_layer.coder.decode(y)

                # Check output
                correct += (out == end)
                total += 1
        return float(correct) / total

    def test(self, mappings):
        return {
            "new_acc" : self.test_traversal(mappings),
            "rec_acc" : self.test_recovery(mappings) }

def print_results(prefix, results):
    print(
        ("%5s" % prefix) +
        " ".join("%10.4f" % results[k]
            for k in [
                "new_acc", "rec_acc" ]))



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
print("     " + " ".join("%10s" % x for x in ["new_acc", "rec_acc"]))

mappings = gen_mappings(num_states, num_inputs, num_trans)
print_results("", test(N, pad, mask_frac, mappings))
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
