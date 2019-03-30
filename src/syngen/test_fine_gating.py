import sys
sys.path.append('../nvm')

from random import randint

from layer import Layer
from activator import tanh_activator
from coder import Coder
from learning_rules import rehebbian

import numpy as np

# TODO:
#
# 1. Mask synapses instead of neurons
#    a. To subset, with subsequent pattern completion
#    b. To full pattern directly
#
# XXX 2. Include recurrent activity in learning (beyond gating)
#    -> g * tanh(W*v_B + W*v_A)
#
# 3. Make masks compositional
#    -> recurrent X input
#
# ??? 4. Make mask based on input instead of state

def test(N, pad, mask_frac, mappings, stabil=5):
    fsm_states = mappings.keys()
    input_states = list(set(k for m in mappings.values() for k,v in m))

    shape = (N,N)
    size = N**2

    act = tanh_activator(pad, size)
    input_layer, fsm_layer = (Layer(k, shape, act, Coder(act)) for k in "ab")
    input_layer.encode_tokens(input_states, orthogonal=True)
    fsm_layer.encode_tokens(fsm_states, orthogonal=True)

    ########### OLD METHOD ###################

    # Learn weight matrices
    w = np.zeros((size,size*2))
    b = np.zeros((size,1))
    for s,m in mappings.items():
        X = input_layer.encode_tokens([k for k,v in m])
        s = np.repeat(fsm_layer.coder.encode(s), X.shape[1], axis=1)
        X = np.concatenate((X, s), axis=0)

        Y = fsm_layer.encode_tokens([v for k,v in m])
        dw,db = rehebbian(w, b, X, Y, act, act)
        w = w + dw

    # Test
    correct = 0
    weighted = 0.
    total = 0
    for start,m in mappings.items():
        start = fsm_layer.coder.encode(start)

        for inp,end in m:
            x = np.concatenate((input_layer.coder.encode(inp), start), axis=0)
            y = act.f(w.dot(x))
            out = fsm_layer.coder.decode(y)

            if out == end:
                correct += 1
                weighted += 1.0
            else:
                weighted += float(len(np.where(
                    np.sign(y) == np.sign(fsm_layer.coder.encode(end))))) / size
            total += 1
    old_acc = float(correct) / total
    weighted_old_acc = weighted / total


    ########### NEW METHOD ###################

    input_layer, fsm_layer = (Layer(k, shape, act, Coder(act)) for k in "ab")
    input_layer.encode_tokens(input_states, orthogonal=False)
    fsm_layer.encode_tokens(fsm_states, orthogonal=False)

    # Create gating masks for each state
    masks = {
        s: (np.random.random(shape) < (1. / mask_frac)
                ).astype(np.float).reshape(-1,1)
            for s in fsm_states }
            #for s in input_states }

    # Ensure nonzero masks
    for mask in masks.values():
        if sum(mask) == 0:
            mask[randint(0, mask.shape[0]-1),0] = 1.

    # Learn recurrent weights
    w_r = np.zeros((size,size))
    b = np.zeros((size,1))
    X = fsm_layer.encode_tokens(fsm_states)
    dw,db = rehebbian(w_r, b, X, X, act, act)
    w_r = w_r + (dw * mask_frac)

    # Test pattern recovery
    correct = 0
    weighted = 0.
    total = 0
    for mask in masks.values():
        for tok in fsm_states:
            complete = fsm_layer.coder.encode(tok)
            partial = np.multiply(complete, mask)
            y = act.f(w_r.dot(partial))

            # Stabilize
            for _ in range(stabil):
                old_y = y
                y = act.f(w_r.dot(y))
                if np.array_equal(y, old_y):
                    break
            out = fsm_layer.coder.decode(y)

            # Check output
            if out == tok:
                correct += 1
                weighted += 1.0
            else:
                weighted += float(len(np.where(
                    np.sign(y) == np.sign(complete)))) / size
            total += 1
    rec_acc = float(correct) / total
    weighted_rec_acc = weighted / total


    # Learn inter-regional weights
    w = np.zeros((size,size))
    b = np.zeros((size,1))
    for start,m in mappings.items():
        # Start state mask, input_layer input
        X = input_layer.encode_tokens([k for k,v in m])
        Y = fsm_layer.encode_tokens([v for k,v in m])

        dw,db = rehebbian(w, b, X, Y, act, act)
        mask = np.repeat(masks[start], size, axis=1)
        w = w + np.multiply(dw, mask)

        '''
        # Input layer mask, start state input
        for k,v in m:
            X = fsm_layer.encode_tokens([start])
            Y = fsm_layer.encode_tokens([v])

            dw,db = rehebbian(w, b, X, Y, act, act)
            mask = np.repeat(masks[k], size, axis=1)
            w = w + np.multiply(dw, mask)
        '''


    # Test
    total = 0
    weighted = 0.
    masked_weighted = 0.
    correct = 0
    masked_correct = 0
    for start,m in mappings.items():
        for inp,end in m:
            # Start state mask, input_layer input
            mask = masks[start]
            x = input_layer.coder.encode(inp)

            '''
            # Input layer mask, start state input
            mask = masks[inp]
            x = fsm_layer.coder.encode(start)
            '''

            y = np.multiply(act.f(w.dot(x)), mask)

            # Check partial match
            end_pat = fsm_layer.coder.encode(end)
            wh = np.where(np.multiply(
                mask, np.sign(y) != np.sign(end_pat)) > 0.)

            if len(wh[0]) == 0:
                masked_correct += 1
                masked_weighted += 1.0
            else:
                mask_size = sum(mask > 0.)
                masked_weighted += (mask_size - len(wh[0])) / mask_size

            # Stabilize
            for _ in range(stabil+1):
                old_y = y
                y = act.f(w_r.dot(y))
                if np.array_equal(y, old_y):
                    break
            out = fsm_layer.coder.decode(y)

            # Check output
            if out == end:
                correct += 1
                weighted += 1.0
            else:
                weighted += float(len(np.where(
                    np.sign(y) == np.sign(fsm_layer.coder.encode(end))))) / size
            total += 1
    new_acc = float(correct) / total
    masked_acc = float(masked_correct) / total
    weighted_new_acc = weighted / total
    weighted_masked_acc = masked_weighted / total

    return {
        "old_acc" : old_acc,
        "new_acc" : new_acc,
        "rec_acc" : rec_acc,
        "masked_acc" : masked_acc,
        "weighted_old_acc" : weighted_old_acc,
        "weighted_new_acc" : weighted_new_acc,
        "weighted_rec_acc" : weighted_rec_acc,
        "weighted_masked_acc" : weighted_masked_acc }

def print_results(prefix, results):
    print(
        ("%5s" % prefix) +
        " ".join("%10.4f" % results[k]
            for k in [
                "old_acc", "new_acc", "rec_acc", "masked_acc" ])
        + " | " +
        " ".join("%10.4f" % results[k]
            for k in [
                "weighted_old_acc", "weighted_new_acc",
                "weighted_rec_acc", "weighted_masked_acc" ])
    )



def gen_mappings(num_states, num_inputs, num_trans):
    # Create finite state machine with input conditional transitions
    fsm_states = [str(x) for x in range(num_states)]
    input_tokens = [str(x) for x in range(num_inputs)]

    # Encode transitions
    mappings = dict()
    for f in fsm_states:
        others = [x for x in fsm_states if x != f]
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


# Parameters
N = 16
pad = 0.0001

print("N=%d" % N)
print("")

mask_frac = N
num_states = N / 2 + 1
num_inputs = N
num_trans = N / 4

fracs = [4, 8, 16, 32, 64, 128]
ns = [4, 8, 16, 32]

res = [[dict() for y in ns] for x in fracs]
masked_res = [[0 for y in ns] for x in fracs]
for i,x in enumerate(fracs):
    for j,y in enumerate(ns):
        mappings = gen_mappings(y,y-1,y-1)
        r = test(N, pad, x, mappings)
        for k,v in r.items():
            res[i][j][k] = v

keys = [
    "old_acc",
    "new_acc",
    "rec_acc",
    "masked_acc",
    "weighted_old_acc",
    "weighted_new_acc",
    "weighted_rec_acc",
    "weighted_masked_acc"
]
for key in keys:
    print(key)
    print("     " + " ".join("%6d" % x for x in ns))
    for i,row in enumerate(res):
        print(" ".join(["%4d" % fracs[i]] + ["%6.4f" % x[key] for x in row]))
    print("")


print("     " + " ".join("%10s" % x for x in ["old_acc", "new_acc", "rec_acc", "masked_acc" ]))

# mask_frac should be equal to N
# redundancy in transition inputs should be minimized
#
# num_trans < num_states
# num_trans < num_inputs

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
print("")

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
    mappings = gen_mappings(num_states, num_inputs, num_trans)
    print_results(x, test(N, pad, x, mappings))
print("")

print("num_states")
for x in [N/2, N, N*2]:
    mappings = gen_mappings(x, num_inputs, num_trans)
    print_results(x, test(N, pad, mask_frac, mappings))
print("")

print("num_inputs")
for x in [N/4, N/2, N, N*2]:
    mappings = gen_mappings(num_states, x, num_trans)
    print_results(x, test(N, pad, mask_frac, mappings))
print("")

print("num_trans")
for x in [N/8, N/4, N/2]:
    mappings = gen_mappings(num_states, num_inputs, x)
    print_results(x, test(N, pad, mask_frac, mappings))
print("")
