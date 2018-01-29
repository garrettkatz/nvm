import matplotlib.pyplot as plt
import numpy as np

w_ii = 1.6
a = np.linspace(-1,1,200)
a = a[1:-1]
z = np.arctanh(a) - w_ii*a
plt.plot(a, np.zeros(z.shape),'--k')
plt.plot([0,0],z[[0,-1]],'--k')
plt.plot(a,z,'-k')
plt.xlim(a[[0,-1]])
plt.ylim(z[[0,-1]])
plt.show()
