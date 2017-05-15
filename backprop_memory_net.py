import numpy as np
import memory_utils as mu

class BackPropKVNet:
    def __init__(self, N, f, df, af):
        # N[k]: size of k^th layer (len L+1)
        # f[k]: activation function of k^th layer (len L)
        # df[k]: derivative of f[k] (len L)
        # af[k]: inverse of f[k] (len L)
        self.L = len(N)-1
        self.N = N
        self.f = f
        self.df = df
        self.af = af
        self.W = mu.init_randn_W(N)
        # for passive ticks
        self.last_key = self.first()
        self.current_key = self.first()
    def layer_size(self):
        return self.N[0]
    def read(self, key):
        return np.sign(mu.forward_pass(key, self.W, self.f)[self.L])
    def write(self, key, value, eta=0.01, num_epochs=1000, verbose=0):
        # train k -> v with gradient descent
        for epoch in range(num_epochs):
            x = mu.forward_pass(key, self.W, self.f)
            # early termination:
            if (np.fabs(x[self.L] - value) < .5).all(): break
            # Progress update:
            e = x[self.L] - value
            if epoch % int(num_epochs/10) == 0:
                E = 0.5*(e**2).sum()
                if verbose > 0: print('%d: %f'%(epoch,E))
                if verbose > 1:
                    print(x[self.L].T)
                    print(value.T)
            # Weight update:
            y = mu.backward_pass(x, e, self.W, self.df)
            G = mu.error_gradient(x, y)
            for k in self.W:
                self.W[k] += - eta * G[k]
    def next(self, key):
        next_key = mu.int_to_pattern(self.layer_size(), mu.patterns_to_ints(key)[0]+1)
        self.last = next_key
        return next_key
    def first(self):
        return mu.int_to_pattern(self.layer_size(), 0)
    def passive_ticks(self, num_ticks, eta=0.01):
        for t in range(num_ticks):
            # self.write(self.current_key, np.sign(self.read(self.current_key)))
            x = mu.forward_pass(self.current_key, self.W, self.f)
            value = np.sign(x[self.L])
            # Progress update:
            e = x[self.L] - value
            # Weight update:
            y = mu.backward_pass(x, e, self.W, self.df)
            G = mu.error_gradient(x, y)
            for k in self.W:
                self.W[k] += - eta * G[k]
            # passive advance
            if (self.current_key == self.last_key).all():
                self.current_key = self.first()
            else:
                self.current_key = self.next(self.current_key)

def make_tanh_kvn(layer_sizes):
    N = layer_sizes
    L = len(N)-1
    f, df, af = [np.tanh]*L, [mu.tanh_df]*L, [np.arctanh]*L
    f = {k: np.tanh for k in range(1,L+1)}
    df = {k: mu.tanh_df for k in range(1,L+1)}
    af = {k: np.arctanh for k in range(1,L+1)}
    kvn = BackPropKVNet(N=N,f=f,df=df,af=af)
    return kvn


if __name__=="__main__":
    N = [3]*3
    L = len(N)-1
    f, df, af = [np.tanh]*L, [mu.tanh_df]*L, [np.arctanh]*L
    f = {k: np.tanh for k in range(1,L+1)}
    df = {k: mu.tanh_df for k in range(1,L+1)}
    af = {k: np.arctanh for k in range(1,L+1)}
    kvn = BackPropKVNet(N=N,f=f,df=df,af=af)
    k, v = np.sign(np.random.randn(N[0],1)), np.sign(np.random.randn(N[0],1))
    kvn.write(k,v)
    print('target vs read')
    print(v.T)
    print(kvn.read(k).T)

class KVNet:
    """
    Feed-forward net for neural key-value memory
    """
    def __init__(self, N, f, df, af):
        # N[k]: size of k^th layer (len L+1)
        # f[k]: activation function of k^th layer (len L)
        # df[k]: derivative of f[k] (len L)
        # af[k]: inverse of f[k] (len L)
        self.L = len(N)-1 #
        self.N = N
        self.f = f
        self.df = df
        self.af = af
        # self.W = [np.random.randn(N[k+1],N[k])/(N[k+1]*N[k]) for k in range(self.L)]
        self.W = [np.eye(N[k+1],N[k]) for k in range(self.L)]
        self.O = [np.zeros((N[k+1],N[k])) for k in range(self.L)]
    def forward_pass(self, x_0):
        # x_0: input layer activity pattern
        x = {0: x_0}
        for k in range(self.L):
            x[k+1] = self.f[k]((self.W[k]+self.O[k]).dot(x[k]))
        return x
    def backward_pass(self, x, d):
        # x[k]: k^th layer activity pattern from forward pass
        # d: target pattern for L^th layer
        k = self.L
        e = (x[k] - d)
        y = {}
        for k in range(self.L,0,-1):
            y[k] = self.df[k-1]((self.W[k-1]+self.O[k-1]).dot(x[k-1])) * e
            e = (self.W[k-1]+self.O[k-1]).T.dot(y[k])
        return y
    def error_gradient(self, x, y):
        # x,y from forward/backward pass
        # returns error gradient wrt omega
        return [y[k+1] * x[k].T for k in range(self.L)]
    def gradient_descent(self, x_0, d, eta=0.01, num_epochs=10000, verbose=2):
        for k in range(L): self.O[k] *= 0.
        for epoch in range(num_epochs):
            x = self.forward_pass(x_0)
            y = self.backward_pass(x, d)
            if epoch % int(num_epochs/10) == 0:
                E = 0.5*((x[self.L]-d)**2).sum()
                if verbose > 0: print('%d: %f'%(epoch,E))
                if verbose > 1:
                    print(x[self.L].T)
                    print(d.T)
            G = self.error_gradient(x, y)
            for k in range(self.L):
                kvn.O[k] += - eta * G[k]
        # Persist weight changes
        for k in range(self.L):
            kvn.W[k] += kvn.O[k]
            kvn.O[k] *= 0
    def memorize(self, x_0, d, eta=0.01, num_epochs=5000, dot_term=0.999, verbose=1):
        x = self.forward_pass(x_0)
        for k in range(L-1): self.O[k] *= 0.
        self.O[L-1]  = (self.af[L-1](d) - self.W[L-1].dot(x[L-1])) * x[L-1].T / (x[L-1].T.dot(x[L-1]))
        learning_curve = []
        epoch = 0
        while True:
            # progress
            x = self.forward_pass(x_0)
            y = self.backward_pass(x,d)
            g = self.error_gradient(x,y)
            E = 0.5*((x[self.L]-d)**2).sum()
            O = sum([(self.O[k]**2).sum() for k in range(self.L)])
            G = sum([(g[k]**2).sum() for k in range(self.L)])
            GO = sum([(g[k]*self.O[k]).sum() for k in range(self.L)])
            if epoch % int(np.ceil(num_epochs/10. + 1)) == 0:
                # P = sum([(g[k]*self.O[k]).sum() for k in range(self.L)])
                if verbose > 0:
                    term = GO**2/(G*O) if G*O > 0 else 0
                    print('%d: E=%f,O=%f,term=%f'%(epoch,E,O,term))
                if verbose > 1:
                    print(x[self.L].T)
                    print(d.T)
            learning_curve.append((E,O,G,GO))
            # termination
            if epoch == num_epochs: break
            if G*O > 0 and GO**2/(G*O) > dot_term: break
            # predictor
            H = GO/G if G > 0 else 0
            for k in range(self.L):
                self.O[k] *= 1 - eta
                self.O[k] += eta * g[k] * H
            # corrector
            x = self.forward_pass(x_0)
            y = self.backward_pass(x, d)
            g = self.error_gradient(x, y)
            for k in range(self.L):
                self.O[k] += - eta * g[k]
            epoch += 1
        # Persist weight changes
        for k in range(self.L):
            kvn.W[k] += kvn.O[k]
            kvn.O[k] *= 0
        return learning_curve
