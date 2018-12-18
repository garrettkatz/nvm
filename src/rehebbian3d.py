import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

np.set_printoptions(linewidth=200)

N = 3
r = .99
r_ = np.arctanh(r)
I = np.eye(N)

# X = np.random.choice([-1,1], (N,N))
# X = -np.ones((N,N)) + 2*I
X = np.array([
    [+1, +1, +1],
    [+1, +1, -1],
    [+1, -1, -1],
    ],dtype=float)

X *= r
g = N * r**2

points = set([(0.,0.,0.,-1,1)])
for expand in range(8):
    print("expansion %d: %d points"%(expand, len(points)))

    point_array = np.array(list(points)).T[:N,:]

    new_points = []
    for c in range(N):
        x = X[:,[c]]
        for pm in [-1,1]:
            new_points.append(
                np.concatenate((
                    (I - x.dot(x.T)/g).dot(point_array) + pm*r_*x/g,
                    c*np.ones((1,len(points))),
                    pm*np.ones((1,len(points)))
                    ), axis=0))

    new_points = np.concatenate(new_points, axis=1)
    new_points = np.round(new_points, decimals=6)
    new_points = set([tuple(p) for p in new_points.T])
    
    points |= new_points

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect("equal")

for c in range(N):
    ax.plot(*zip([0, 0, 0], X[:,c]), linestyle='-', color='k')

point_array = np.array(list(points)).T
points, point_groups, point_signs = point_array[:N,:], point_array[N,:], point_array[N+1,:]
for c in range(N):
# for c in [0]:
    for pm in [-1,1]:

        points_c = points[:,(point_groups==c) & (point_signs==pm)]
        print(np.fabs(X[:,[c]].T.dot(points_c) - pm*r_).max())
    
        ax.plot(*points_c, linestyle='none', color='rbg'[c], marker='.')

ax.set_aspect("equal")
# plt.axis("equal")

plt.show()

