"""
Implementation of GALIS neural network model for sequence learning
"""

class GALISNN:
    def __init__(self, network_size, k, k_theta, k_w, history=2):
        self.N = network_size
        self.k = k
        self.k_theta = k_theta
        self.k_w = k_w
        self.W = np.zeros((self.N, self.N))
        self.V = np.zeros((self.N, self.N))
        self.a
        self.theta
        self.history = history
        self.tick_mark = 0
    def activate(self, external_input):
        theta = self.a[:,[self.tick_mark]]
        theta = (1-self.k_theta)*theta
        a = self.a[:,[self.tick_mark]]
        h = self.V.dot(a) - theta + external_input
        a = np.sign(h)
        theta = (1-self.k_theta)*theta
        f = self.W.dot(a) - theta + external_input
        a = np.sign(f)
        theta = (1-self.k_theta)*theta
    def learn(self, a_old, a_new):
        """
        Temporally associates activation patterns a_old and a_new
        """
        self.W = (1-self.k)*self.W + (1/self.N)*a_new*a_new.T*(1-np.eye(self.N))
        self.V = (1-self.k)*self.V + (1/self.N)*a_old*a_new.T
