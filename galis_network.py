"""
Implementation of GALIS neural network model for sequence learning
(Sylvester 2008)
Confusion about 2016 (ISM):
for V, uses a_old*a_new.T, shouldn't it be a_new*a_old.T?
confusing that the same t is used, but multiple stages in the update process.  is theta updated three times, before h, between h and f, and after f?  or only two?  and when? are all three of these thetas referred to as theta(t)?
"""

class GALISNN:
    def __init__(self, N, k_d, k_theta, k_w, beta_1, beta_2, history=2):
        self.N = N
        self.k_d = k_d
        self.k_theta = k_theta
        self.k_w = k_w
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.history = history
        self.W = np.zeros((self.N, self.N))
        self.V = np.zeros((self.N, self.N))
        self.a = np.zeros((self.N, self.history))
        self.theta = np.zeros((self.N, self.history))
        self.t = 0
    def activate(self):
        """
        Update the network activations
        """
        a = self.a[:,[self.t]]
        theta = self.theta[:,[self.t]]
        h = self.beta_1*self.W.dot(a) + self.beta_2*self.V.dot(a) - theta
        a_new = np.sign(h) + (h==0)*a
        theta_new = (1-self.k_theta)*theta + (a_new==a)*self.k_w*a
        self.t = (self.t + 1) % self.history
        self.a[:,[self.t]] = a_new
        self.theta[:,[self.t]] = theta_new
    def associate(self):
        """
        Temporally associate past two activity patterns
        """
        a_new = self.a[:,[self.t]]
        a_old = self.a[:,[(self.t - 1) % self.history]]
        self.W = (1-self.k_d)*self.W + (1/self.N)*a_new*a_new.T*(1-np.identity(self.N))
        self.V = (1-self.k_d)*self.V + (1/self.N)*a_new*a_old.T


if __name__ == '__main__':
    pass
