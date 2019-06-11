"""
Used to generate Figure 10 in the 2019 NVM paper
"""
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
# X = np.array([
#     [+1, +1, +1],
#     [+1, +1, -1],
#     [+1, -1, -1],
#     ],dtype=float)

X = (-1.) ** np.array([(np.arange(2**N) / 2**m) % 2 for m in range(N)])
print(X)
# raw_input('.')

X *= r
g = N * r**2

points = set([(0.,0.,0.,-1,1)])
points_e = []
expand = 7
norm_stats = []
for e in range(expand):
    print("expansion %d: %d points"%(e, len(points)))

    point_array = np.array(list(points)).T[:N,:]

    new_points = []
    for c in range(X.shape[1]):
        x = X[:,[c]]
        for pm in [-1,1]:
            new_points.append(
                np.concatenate((
                    (I - x.dot(x.T)/g).dot(point_array) + pm*r_*x/g,
                    c*np.ones((1,len(points))),
                    pm*np.ones((1,len(points)))
                    ), axis=0))

    new_points = np.concatenate(new_points, axis=1)

    # new_points = np.round(new_points, decimals=9)
    new_points = set([tuple(p) for p in new_points.T])
    
    points |= new_points
    
    points_e.append(set(points))


    point_array = np.array(list(points)).T[:N,:]
    norms2 = np.sqrt((point_array[:N,:]**2).sum(axis=0))
    print(norms2.shape)
    norm_stats.append((norms2.min(), norms2.mean(), norms2.std(), norms2.max()))

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(121, projection='3d', proj_type='ortho')
ax.set_aspect("equal")

# for c in range(X.shape[1]):
#     ax.plot(*zip([0, 0, 0], X[:,c]), linestyle='-', color='k')

points_e = points_e[::-1]
for e in range(expand):

    point_array = np.array(list(points_e[e])).T
    points, point_groups, point_signs = point_array[:N,:], point_array[N,:], point_array[N+1,:]
    # for c in range(X.shape[1]):
    for c in [0]:
        for pm in [1]:
        # for pm in [-1,1]:
    
            points_c = points[:,(point_groups==c) & (point_signs==pm)]
            print(np.fabs(X[:,[c]].T.dot(points_c) - pm*r_).max())
        
            # ax.plot(*points_c, linestyle='none', color='rbggggggggggggggggg'[c], marker='.')
            ax.plot(*points_c, linestyle='none', color=((e+1)/float(expand),)*3, marker='.')

# a = r_ / X[:,[1,2,3,1]].T.dot(X[:,[0]])
# w = X[:,[1,2,3,1]] * a.T
# ax.plot(*w, linestyle='-', color='r')

# x = -X[:,[1,2,3,1]]
# x = x - 2*X[:,[0]].dot(X[:,[0]].T).dot(x)/N
# a = r_ / x.T.dot(X[:,[0]])
# w = x * a.T
# ax.plot(*w, linestyle='-', color='r')

# a = r_ / X[:,[0,2,3,0]].T.dot(X[:,[1]])
# w = X[:,[0,2,3,0]] * a.T
# ax.plot(*w, linestyle='-', color='b')

# x = -X[:,[0,2,3,0]]
# x = x - 2*X[:,[1]].dot(X[:,[1]].T).dot(x)/N
# a = r_ / x.T.dot(X[:,[1]])
# w = x * a.T
# ax.plot(*w, linestyle='-', color='b')

ax.elev = 36
ax.azim = 45
ax.set_xlim([-1.5,3.5])
ax.set_ylim([-1.5,3.5])
ax.set_zlim([-1.5,3.5])
ax.set_xticks(range(-1,4))
ax.set_yticks(range(-1,4))
ax.set_zticks(range(-1,4))
ax.set_aspect("equal")

ax = fig.add_subplot(122)
mins, means, stds, maxs = zip(*norm_stats)
ax.errorbar(range(len(means)),means, yerr=stds, fmt='ko-')
ax.plot(maxs,'ko--')
plt.legend(["Max","Mean (+/- St. Dev.)"])
plt.xlabel("T")
plt.ylabel("$||w_T||$",rotation=0)
plt.yticks(range(1,5))

plt.tight_layout()

plt.show()

