import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

np.set_printoptions(linewidth=200)

N = 4

# X = (-1.) ** np.array([(np.arange(2**N) / 2**m) % 2 for m in range(N)])
# print(X)
# print(X.dot(X.T))

# # # sub X.T.dot(X) relationships
# # for k in range(N+1):
# #     idx = ((X == -1).sum(axis=0) == k)
# #     x = X[:,idx]
# #     print("")
# #     print("-------")
# #     print("")
# #     print(k)
# #     # print(x)
# #     print(x.dot(x.T))
# #     # print(x.T.dot(x))
# #     print(x.sum(axis=1)[:,np.newaxis])

# r = .99
# r_ = np.arctanh(r)
# I = np.eye(N)

# X *= r
# g = N * r**2

# points = set([(0.,0.,0.,0.,-1,1)])
# for expand in range(4):
#     print("expansion %d: %d points"%(expand, len(points)))

#     point_array = np.array(list(points)).T[:N,:]

#     new_points = []
#     for c in range(X.shape[1]):
#         x = X[:,[c]]
#         for pm in [-1,1]:
#             new_points.append(
#                 np.concatenate((
#                     (I - x.dot(x.T)/g).dot(point_array) + pm*r_*x/g,
#                     c*np.ones((1,len(points))),
#                     pm*np.ones((1,len(points)))
#                     ), axis=0))

#     new_points = np.concatenate(new_points, axis=1)
#     # new_points = np.round(new_points, decimals=9)
#     new_points = set([tuple(p) for p in new_points.T])
    
#     points |= new_points

# point_array = np.array(list(points)).T
# points, point_groups, point_signs = point_array[:N,:], point_array[N,:], point_array[N+1,:]

# x0 = X[:,[0]]
# u, s, vh = np.linalg.svd(I - x0.dot(x0.T)/g)
# B = vh[:N-1,:] # basis of projection space

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
# ax.set_aspect("equal")

# BX = B.dot(X)
# BP = B.dot(points)
# BP0 = B.dot(I - x0.dot(x0.T)/g)

# for c in range(X.shape[1]):
#     ax.plot(*zip([0, 0, 0], BX[:,c]), linestyle='-', color='k')
#     if (X[:,[c]]*x0 >= 0).sum() in [0,N]:
#         ax.plot(*BX[:,[c]], linestyle='none', marker='o',color='r')
#     if (X[:,[c]]*x0 >= 0).sum() in [1,N-1]: 
#         ax.plot(*BX[:,[c]], linestyle='none', marker='o',color='g')
#     if (X[:,[c]]*x0 >= 0).sum() in [2,N-2]: 
#         # ax.plot(*BX[:,[c]], linestyle='none', marker='o',color='b')
#         lam = 2*r_ / (g)
#         ax.plot(*(BX[:,[c]]*lam), linestyle='none', marker='o',color='b')

# for c in range(X.shape[1]):
# # for c in [0,1]:
# #     for pm in [1]:
#     for pm in [-1,1]:

#         points_c = BP[:,(point_groups==c) & (point_signs==pm)]
#         # print(np.fabs(BX[:,[c]].T.dot(points_c) - pm*r_).max())
    
#         # ax.plot(*points_c, linestyle='none', color='rbggggggggggggggggg'[c], marker='.')
#         ax.plot(*points_c, linestyle='none', color='k', marker='.')

# tetra = [0,1,2,0,3,1,2,3]
# a = r_ / X[:,[1,2,4,8]][:,tetra].T.dot(X[:,[0]])
# w = X[:,[1,2,4,8]][:,tetra] * a.T
# # ax.plot(*(B.dot(w)), linestyle='none', color='g', marker='o')
# ax.plot(*(B.dot(w)), linestyle='-', color='g')

# a = r_ / -X[:,[1,2,4,8]][:,tetra].T.dot(-X[:,[0]])
# w = -X[:,[1,2,4,8]][:,tetra] * a.T
# # ax.plot(*(B.dot(w)), linestyle='none', color='g', marker='o')
# ax.plot(*(B.dot(w)), linestyle='-', color='g')

# # x = -X[:,[1,2,3,1]]
# # x = x - 2*X[:,[0]].dot(X[:,[0]].T).dot(x)/N
# # a = r_ / x.T.dot(X[:,[0]])
# # w = x * a.T
# # ax.plot(*w, linestyle='-', color='r')

# # a = r_ / X[:,[0,2,3,0]].T.dot(X[:,[1]])
# # w = X[:,[0,2,3,0]] * a.T
# # ax.plot(*w, linestyle='-', color='b')

# # x = -X[:,[0,2,3,0]]
# # x = x - 2*X[:,[1]].dot(X[:,[1]].T).dot(x)/N
# # a = r_ / x.T.dot(X[:,[1]])
# # w = x * a.T
# # ax.plot(*w, linestyle='-', color='b')

# # ax.set_xlim([-4,4])
# # ax.set_ylim([-4,4])
# # ax.set_zlim([-4,4])

# for c in range(BP0.shape[1]):
#     for pm in [-1,1]:
#         ax.plot(*zip([0, 0, 0], pm*BP0[:,c]*4), linestyle='-', color='k')

# ax.plot(*(+BP0*4), linestyle='none', color='m', marker='o')
# ax.plot(*(-BP0*4), linestyle='none', color='y', marker='o')

# ax.set_aspect("equal")
# # plt.axis("equal")

# plt.show()

data = []
N = 100
P = 10
reps = 20000
for r in range(reps):
    X = np.random.choice([-1,1],size=(N,P))
    p1 = np.random.choice(range(P))
    p2 = np.random.choice(range(P))
    data.append((X[:,p1]*X[:,p2]).sum())

print(len(data))
print(np.mean(data))
print(float(N)/P)
