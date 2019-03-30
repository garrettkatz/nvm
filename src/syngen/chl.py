import sys
sys.path.append('../nvm')

import numpy as np
import pickle

from activator import tanh_activator
from coder import Coder

from op_net import arith_ops

class CHL_Net:
    def __init__(self, Ns, feedback=False, split_learn=False, biases=True):
        self.sizes = [N for N in Ns]
        self.feedback = feedback
        self.split_learn = split_learn
        self.biases = biases

        self.W, self.B = [],[]
        self.G = []
        for i in range(len(Ns)-1):
            size, prior_size = Ns[i+1], Ns[i]
            self.W.append(2 * np.random.random((size, prior_size)) - 1)
            self.B.append(2 * np.random.random((size, 1)) - 1)
            self.G.append(np.random.normal(0.0, 1.0, ((size, prior_size))))

        self.activity = [np.zeros((size,1)) for size in self.sizes]
        self.activity_new = [np.zeros((size,1)) for size in self.sizes]

    def run(self, in_pattern, target=None):
        self.activity[0][:] = in_pattern
        l_range = len(self.sizes)

        # If target output pattern provided, clamp it
        if target is not None:
            self.activity[-1][:] = target
            l_range -= 1

        # Randomly initialize unclamped layers
        for i in range(1,l_range):
            self.activity[i][:] = np.zeros((self.sizes[i],1))

        # If feedback connections are present, allow network to settle
        # Otherwise, make sure information propogates through network
        if self.feedback:
            T = 20
            dt = 0.1
        else:
            T = l_range-1
            dt = 1.0

        for t in range(T):
            for l in range(1,l_range):
                # Output layer doesn't have feedback
                if l == len(self.sizes) - 1:
                    self.activity_new[l][:] = self.activity[l] + dt * (
                        -self.activity[l] + np.tanh(
                           self.W[l-1].dot(self.activity[l-1])
                           + self.B[l-1]))
                elif self.feedback:
                    self.activity_new[l][:] = self.activity[l] + dt * (
                        -self.activity[l] + np.tanh(
                           self.W[l-1].dot(self.activity[l-1])
                           + self.W[l].T.dot(self.activity[l+1])
                           #+ self.G[l].dot(self.activity[l+1])
                           + self.B[l-1]))
                else:
                    self.activity_new[l][:] = self.activity[l] + dt * (
                        -self.activity[l] + np.tanh(
                           self.W[l-1].dot(self.activity[l-1])
                           + self.B[l-1]))

            #if target is None:
            #    print(t, sum((np.abs(self.activity[-2] - self.activity_new[-2])).flat))
            #    print(t, sum((np.abs(self.activity[-1] - self.activity_new[-1])).flat))
            for l in range(1,l_range):
                self.activity[l][:] = self.activity_new[l]

    def test(self, patterns):
        err,correct = (0.0, 0)

        for in_pattern,target in patterns:
            self.run(in_pattern)
            output = self.activity[-1]

            err += sum(((target-output)**2).flat)
            correct += np.all(np.sign(target.flat) == np.sign(output.flat))

        return err, float(correct)/len(patterns)

    def train(self, epochs, patterns, learning_rate=None, verbose=False):
        ada = 1.0 if learning_rate is None else learning_rate

        for epoch in range(epochs):
            for in_pattern,target in patterns:
                # Minus phase
                self.run(in_pattern)

                if np.all(np.sign(target.flat) == np.sign(self.activity[-1].flat)):
                    continue

                if self.split_learn:
                    minus = self.activity
                    for l in range(1,len(self.sizes)):
                        self.W[l-1] -= ada * (minus[l].dot(minus[l-1].T))
                        if self.biases: self.B[l-1] -= ada * minus[l]
                else:
                    minus = [np.copy(act) for act in self.activity]

                # Plus phase
                self.run(in_pattern, target)
                plus = self.activity

                if self.split_learn:
                    for l in range(1,len(self.sizes)):
                        self.W[l-1] += ada * (plus[l].dot(plus[l-1].T))
                        if self.biases: self.B[l-1] += ada * plus[l]
                else:
                    for l in range(1,len(self.sizes)):
                        self.W[l-1] += ada * (
                            plus[l].dot(plus[l-1].T) - minus[l].dot(minus[l-1].T))
                        if self.biases: self.B[l-1] += ada * (plus[l] - minus[l])

            err,acc = self.test(patterns)
            if verbose:
                print("Epoch %4d:   err= %12.4f   acc= %6.4f" % (epoch, err, acc))

            if acc == 1.0: return
            if learning_rate is None:
                ada = 1.0 - acc


### Digits ###
def test_digits(verbose=False):
    epochs = 1000
    pad = 0.0001
    feedback = True
    split_learn = True
    biases = True
    Ns = [256 for _ in range(3)]
    tokens = [str(x) for x in range(100)]

    net = CHL_Net(Ns, feedback, split_learn, biases)

    # Input/output pattern pairs (0-99)
    in_coder = Coder(tanh_activator(pad, (Ns[0])))
    out_coder = Coder(tanh_activator(pad, Ns[-1]))
    patterns = [(in_coder.encode(tok), out_coder.encode(tok)) for tok in tokens]

    net.train(epochs, patterns, verbose=verbose)



### Arithmetic ###
def test_arith(verbose=False):
    epochs = 100
    feedback = False
    split_learn = False
    biases = True
    pad = 0.0001

    N = 256
    Ns = [N * 3, N * 2, N * 2]
    net = CHL_Net(Ns, feedback, split_learn, biases)

    in_size = int(Ns[0]/3)
    out_size = int(Ns[-1]/2)
    in1_coder = Coder(tanh_activator(pad, in_size))
    in2_coder = Coder(tanh_activator(pad, in_size))
    in3_coder = Coder(tanh_activator(pad, in_size))
    out1_coder = Coder(tanh_activator(pad, out_size))
    out2_coder = Coder(tanh_activator(pad, out_size))

    # (x,y,op) => op(x,y) pairs
    patterns = []
    for op in arith_ops:
        for i in range(10):
            for j in range(10):
                in1 = in1_coder.encode(str(i))
                in2 = in2_coder.encode(str(j))
                in3 = in3_coder.encode(op)

                try:
                    f0,f1 = arith_ops[op]
                    out1 = out1_coder.encode(f0(i,j))
                    out2 = out2_coder.encode(f1(i,j))
                except:
                    out1 = out1_coder.encode("null")
                    out2 = out2_coder.encode("null")

                patterns.append((
                    np.append(np.append(in1, in2, axis=0), in3, axis=0),
                    np.append(out1, out2, axis=0)))

    net.train(epochs, patterns, verbose=verbose)


### Gate sequence ###
def test_gate_seq(verbose=False):
    X,Y = pickle.load(open("gate_seq.pkl", "rb"))

    epochs = 100
    feedback = False
    split_learn = False
    biases = True

    Ns = [X.shape[0], 256, Y.shape[0]]
    net = CHL_Net(Ns, feedback, split_learn, biases)

    patterns = tuple(
        (X[:,i].reshape(-1,1), Y[:,i].reshape(-1,1)) for i in range(X.shape[1]))

    net.train(epochs, patterns, verbose=verbose)

if __name__ == "__main__":
    print("\nTesting digits...")
    test_digits(verbose=True)

    print("\nTesting arithmetic...")
    test_arith(verbose=True)

    print("\nTesting gate sequences...")
    test_gate_seq(verbose=True)
