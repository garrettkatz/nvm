import os
from math import isnan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D

tonic = 0.0                                                                     
Ra = 1.0                                                                        
Rv = 20.0 

class Sample:
    def __init__(self, filename, s, av, va, mean_latency, stdev_latency,
            bold_amy, bold_vmpfc, activ_amy, activ_vmpfc, max_activ_amy,
            max_activ_vmpfc, num_responses):
        self.filename = filename
        self.s = s
        self.av = av
        self.va = va
        self.mean_latency = mean_latency
        self.stdev_latency = stdev_latency
        self.bold_amy = bold_amy + ((1.0 - 0.025) * activ_amy)
        self.bold_vmpfc = bold_vmpfc + ((1.0 - 0.00125) * activ_vmpfc)
        self.activ_amy = activ_amy
        self.activ_vmpfc = activ_vmpfc
        self.max_activ_amy = max_activ_amy
        self.max_activ_vmpfc = max_activ_vmpfc
        self.num_responses = num_responses
        self.steady = Ra * (s - va * tonic) / (1 + (Ra * Rv * av * va)) 

    def valid(self, num_faces):
        return num_faces == self.num_responses

    def clipped(self, threshold):
        return self.mean_latency == threshold

    def to_3d(self):
        return (self.s, self.av, self.va)

    def cls(self, num_faces, threshold):
        v = self.valid(num_faces)
        c = self.clipped(threshold)
        return (v and not c, not v, c).index(True)

    def activ_bounded(self, bound):
        return self.max_activ_amy < bound and self.max_activ_vmpfc < bound

    def __str__(self):
        return (" %20s: " + ("%8.4f " * 5)) % (
            self.filename, self.s, self.av, self.va, self.mean_latency, self.steady)

    def __eq__(self, other): 
        return self.s == other.s \
            and self.av == other.av \
            and self.va == other.va \
            and self.mean_latency == other.mean_latency \
            and self.stdev_latency == other.stdev_latency \
            and self.bold_amy == other.bold_amy \
            and self.bold_vmpfc == other.bold_vmpfc \
            and self.activ_amy == other.activ_amy \
            and self.activ_vmpfc == other.activ_vmpfc \
            and self.max_activ_amy == other.max_activ_amy \
            and self.max_activ_vmpfc == other.max_activ_vmpfc \
            and self.num_responses == other.num_responses

samples = []
max_samples = 99999

#for d in ("explore_trial_2",): #"healthy", "ptsd"):
for d in ("explore_trial_2", "explore"):
    for f in sorted(os.listdir(d)):
        if f.endswith(".txt"):
            count = len(tuple(l for l in open("%s/%s" % (d, f)) if "correct" in l))

            s = 0.0
            av = 0.0
            va = 0.0
            mean_latency = 0.0
            stdev_latency = 0.0
            bold_amy = 0.0
            bold_vmpfc = 0.0
            activ_amy = 0.0
            activ_vmpfc = 0.0
            max_activ_amy = 0.0
            max_activ_vmpfc = 0.0
            num_responses = count

            for line in open("%s/%s" % (d, f)):
                if line.startswith("Using amygdala sensitivity:"):
                    s = (float(line.split(":")[1].strip()))
                elif line.startswith("Response latency mean:"):
                    mean_latency = (float(line.split(":")[1].strip()))
                elif line.startswith("Response latency std dev:"):
                    stdev_latency = (float(line.split(":")[1].strip()))
                elif line.startswith("Using amygdala->vmPFC:"):
                    av = (float(line.split(":")[1].strip()))
                elif line.startswith("Using vmPFC->amygdala:"):
                    va = (float(line.split(":")[1].strip()))
                elif line.startswith("BOLD Amygdala:"):
                    bold_amy = (float(line.split(":")[1].strip()))
                elif line.startswith("BOLD vmPFC:"):
                    bold_vmpfc = (float(line.split(":")[1].strip()))
                elif line.startswith("Activation Amygdala:"):
                    activ_amy = (float(line.split(":")[1].strip()))
                elif line.startswith("Activation vmPFC:"):
                    activ_vmpfc = (float(line.split(":")[1].strip()))
                elif line.startswith("Max Activation Amygdala:"):
                    max_activ_amy = (float(line.split(":")[1].strip()))
                elif line.startswith("Max Activation vmPFC:"):
                    max_activ_vmpfc = (float(line.split(":")[1].strip()))

            if mean_latency != 0.0:
                samples.append(Sample(f, s, av, va, mean_latency, stdev_latency,
                    bold_amy, bold_vmpfc, activ_amy, activ_vmpfc, max_activ_amy,
                    max_activ_vmpfc, num_responses))
                if len(samples) == max_samples:
                    break


valid = tuple(s for s in samples if s.valid(5))
invalid = tuple(s for s in samples if not s.valid(5))
clipped = tuple(s for s in samples if s.clipped(67.0))
good = tuple(s for s in samples if s.valid(5) and not s.clipped(67.0))

print("Valid: %d" % len(valid))
print("Invalid: %d" % len(invalid))
print("Clipped: %d" % len(clipped))
print("Good: %d" % len(good))

'''
print("\nClipped:")
for s in clipped: print(s)

print("\nGood:")
for s in sorted(good, key=lambda x: x.mean_latency): print(s)

print("\nInvalid:")
for s in invalid: print(s)
'''

mean_s = np.mean(tuple(sample.s for sample in good))
mean_mean_latency = np.mean(tuple(sample.mean_latency for sample in good))
mean_stdev_latency = np.mean(tuple(sample.stdev_latency for sample in good))
mean_av = np.mean(tuple(sample.av for sample in good))
mean_va = np.mean(tuple(sample.va for sample in good))
mean_bold_amy = np.mean(tuple(sample.bold_amy for sample in good))
mean_bold_vmpfc = np.mean(tuple(sample.bold_vmpfc for sample in good))
mean_activ_amy = np.mean(tuple(sample.activ_amy for sample in good))
mean_activ_vmpfc = np.mean(tuple(sample.activ_vmpfc for sample in good))

stdev_amy = np.std(tuple(sample.s for sample in good))
stdev_mean = np.std(tuple(sample.mean_latency for sample in good))
stdev_stdev = np.std(tuple(sample.stdev_latency for sample in good))

# ms per timestep
scale_factor = (440.0 - 120.0) / mean_mean_latency

print("\n%s condition:" % d)
print("Mean Amygdala sensitivity:")
print(mean_s)
print("")
print("Mean amygdala -> vmPFC strength:")
print(mean_av)
print("")
print("Mean vmPFC -> amygdala strength:")
print(mean_va)
print("")
print("Mean latency:")
print(mean_mean_latency, mean_mean_latency * scale_factor,
    mean_mean_latency * scale_factor + 120)
print("")
print("Mean std dev latency:")
print(stdev_mean, stdev_mean * scale_factor)
print("")
print("Mean BOLD amygdala:")
print(mean_bold_amy)
print("")
print("Mean BOLD vmPFC:")
print(mean_bold_vmpfc)
print("")
print("Mean activation amygdala:")
print(mean_activ_amy)
print("")
print("Mean activation vmPFC:")
print(mean_activ_vmpfc)
print("\n\n")

fig = plt.figure()

mean_latencies = tuple(sample.mean_latency for sample in good)
bold_vmpfcs = tuple(sample.bold_vmpfc for sample in good)
max_activ_amys = tuple(sample.max_activ_amy for sample in good)
max_activ_vmpfcs = tuple(sample.max_activ_vmpfc for sample in good)

# Plot latencies and BOLD
plt.subplot(221)
plt.title("Mean latencies")
plt.hist(mean_latencies)
plt.subplot(222)
plt.title("BOLD vmPFCs")
plt.hist(bold_vmpfcs)
plt.subplot(223)
plt.title("Max Amygdala Activations")
plt.hist(max_activ_amys)
plt.subplot(224)
plt.title("Max vmPFC Activations")
plt.hist(max_activ_vmpfcs)
plt.show()

fig = plt.figure()

# Perform linear regression to predict latencies in good samples
data = [s.to_3d() for s in good]
labels = mean_latencies

model = LinearRegression()
model.fit(data, labels)
print("Latency model",
    model.coef_, model.intercept_,
    model.score(data, labels),
    model.predict([(0.4, 1.0, 1.0), (0.6, 0.25, 2.5)]))

# Plot data points by class
ax = fig.add_subplot(131, projection='3d')
ax.set_title("Full dataset")

x, y, z = zip(*[s.to_3d() for s in good])
ax.scatter(x, y, z, c='yellow')

x, y, z = zip(*[s.to_3d() for s in invalid])
ax.scatter(x, y, z, c='red')

x, y, z = zip(*[s.to_3d() for s in clipped])
ax.scatter(x, y, z, c='green')

# Add spheres for healthy/PTSD
ax.scatter([0.4, 0.6], [1.0, 0.25], [1.0, 2.5], s=1000, c='cyan')


# Perform linear regression to predict latencies in good samples
data = [s.to_3d() for s in good]
labels = bold_vmpfcs

model = LinearRegression()
model.fit(data, labels)
print("vmPFC BOLD coefficients",
    model.coef_, model.intercept_,
    model.score(data, labels),
    model.predict([(0.4, 1.0, 1.0), (0.6, 0.25, 2.5)]))

# Perform logistic regression to separate valid, invalid, and clipped
data = tuple(s.to_3d() for s in samples)
labels = tuple(s.cls(5, 67.0) for s in samples)

x_train, x_test, y_train, y_test = \
    train_test_split(data, labels, test_size=0.25, random_state=0)

model = LogisticRegression()
model.fit(x_train, y_train)
print("Logistic model:",
    model.coef_, model.intercept_, model.n_iter_,
    model.score(x_test, y_test))

print("Confusion matrix (good, invalid, clipped):")
print(confusion_matrix(
    np.array(y_test),
    np.array(model.predict(x_test))))

for i in (1,2):
    xx, yy, zz = np.mgrid[0:2:.05, 0:5:.05, 0:5:.05]
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    probs = model.predict_proba(grid)[:, i]
    grid = [grid[i] for i in xrange(len(probs)) if probs[i] > 0.499 and probs[i] < 0.501]
    probs = [p for p in probs if p > 0.499 and p < 0.501]

    xx, yy, zz = zip(*grid)
    #ax.scatter(xx, yy, zz)
    ax.plot_trisurf(xx, yy, zz)


# Plot latency of good data points using heatmap
ax = fig.add_subplot(132, projection='3d')
ax.set_title("Latency Heatmap")

data = np.array([s.to_3d() for s in good])
labels = np.array([s.mean_latency for s in good])
x, y, z = zip(*data)
p = ax.scatter(x, y, z, c=labels, cmap='gist_heat')
fig.colorbar(p)

# Add spheres for healthy/PTSD
ax.scatter([0.4, 0.6], [1.0, 0.25], [1.0, 2.5], s=1000, c='cyan')


# Plot vmPFC bold of good data points using heatmap
ax = fig.add_subplot(133, projection='3d')
ax.set_title("vmPFC BOLD Heatmap")

data = np.array([s.to_3d() for s in good])
labels = np.array([s.bold_vmpfc for s in good])
x, y, z = zip(*data)
p = ax.scatter(x, y, z, c=labels, cmap='gist_heat')
fig.colorbar(p)

# Add spheres for healthy/PTSD
ax.scatter([0.4, 0.6], [1.0, 0.25], [1.0, 2.5], s=1000, c='cyan')

plt.tight_layout()
plt.show()


# SEARCH PARAMS
vmpfc_activ_bound = 0.5
bins = 25

latency_deviation_cutoff = 0.05
bold_ratio_min = 0.25
bold_ratio_max = 0.75

param_ratio_filter = False
param_ratio_min = 0.25
param_ratio_max = 4.0


# Bin the data and search for increased latency and decreased vmPFC bold
# Use only data where max vmPFC activation stays below upper bound
bounded = [s for s in good if s.activ_bounded(vmpfc_activ_bound)]
data = np.array([[sample.s, sample.av, sample.va]
    for sample in bounded]).reshape(len(bounded), 3)
hist, binedges = np.histogramdd(data, normed=False, bins=bins)
bins_s, bins_av, bins_va = binedges

# Place each sample into a bin
indices = set()
for sample in bounded:
    i_s = 0
    i_av = 0
    i_va = 0

    for i,b in enumerate(bins_s):
        if sample.s > b:
            i_s = i

    for i,b in enumerate(bins_av):
        if sample.av > b:
            i_av = i

    for i,b in enumerate(bins_va):
        if sample.va > b:
            i_va = i

    indices.add((i_s, i_av, i_va))

#good = [c for c in good if c.mean_latency > 100]
binned = dict((index, []) for index in indices)
for sample,index in zip(bounded, indices):
    binned[index].append(sample)

# Functions for processing parameters
def get_params(i):
    return(
        (bins_s[i[0]+1] + bins_s[i[0]]) / 2,
        (bins_av[i[1]+1] + bins_av[i[1]]) / 2,
        (bins_va[i[2]+1] + bins_va[i[2]]) / 2)
def get_param_ratios(l, r):
    l_s, l_av, l_va = get_params(l)
    r_s, r_av, r_va = get_params(r)
    return (r_s / l_s, r_av / l_av, r_va / l_va)
def polarize_ratios(ratios):
    a,b,c = ratios
    a_adj = a - 1.0 if a > 1.0 else -(1.0 / a - 1.0)
    b_adj = b - 1.0 if b > 1.0 else -(1.0 / b - 1.0)
    c_adj = c - 1.0 if c > 1.0 else -(1.0 / c - 1.0)
    return a_adj, b_adj, c_adj

# Compute mean of latencies and vmPFC BOLD for each bin
latencies = dict()
bolds = dict()
for key,val in binned.iteritems():
    if len(val) > 0:
        latencies[key] = sum(x.mean_latency for x in val) / len(val)
        bolds[key] = sum(x.bold_vmpfc for x in val) / len(val)

# Find parameter set pairs where:
# 1. Amygdala sensitivity increases
# 2. Saccade latency increases
# 3. vmPFC BOLD decreases
candidates = []
for l in indices:
    for r in indices:
        sensitivity_ratio = get_params(r)[0] / get_params(l)[0]
        latency_ratio = latencies[r] / latencies[l]
        bold_ratio = bolds[r] / bolds[l]

        if sensitivity_ratio > 1.0 and latency_ratio > 1.0 and bold_ratio < 1.0:
            candidates.append((l,r, latency_ratio, bold_ratio))

print("\nFound %d / %d candidate parameter pairs" % \
    (len(candidates), len(indices) ** 2 / 2 - len(indices)))


# Filter out candidates by deviation from ideal latency
target = 1.603125
candidates = [c for c in candidates if abs(target - c[2]) / target < latency_deviation_cutoff]
print("    ... %d within one percent of target latency ratio" % len(candidates))

# Filter out candidates by BOLD ratio cutoff
candidates = [c for c in candidates if c[3] > bold_ratio_min and c[3] < bold_ratio_max]
print("    ... %d with < 0.5 vmPFC BOLD ratio" % len(candidates))

# Filter out any candidates < 0.5 or > 2.0 parameter ratios
if param_ratio_filter:
    candidates = [c for c in candidates
        if all(r > param_ratio_min and r < param_ratio_max
                for r in get_param_ratios(c[0], c[1]))]
    print("    ... %d with 0.5 < r < 2.0 for all parameter ratios" % len(candidates))

# Sort by BOLD ratio
candidates = sorted(candidates, key = lambda x: x[3])

print("\nParameter set candidates:")
for l,r,lat_rat,bold_rat in candidates:
    print("=" * 80)
    l_s, l_av, l_va = get_params(l)
    print(("%9.4f " * 5) % (l_s, l_av, l_va, latencies[l], bolds[l]))

    r_s, r_av, r_va = get_params(r)
    print(("%9.4f " * 5) % (r_s, r_av, r_va, latencies[r], bolds[r]))

    print("-" * 80)
    print(("%9.4f " * 5) % (r_s / l_s, r_av / l_av, r_va / l_va, lat_rat, bold_rat))
    print("")

print("\nTotal candidates: %d" % len(candidates))

# Plot parameter ratios colored by BOLD ratios
param_ratios = []
bold_ratios = []
for l,r,lat_rat,bold_rat in candidates:
    param_ratios.append(get_param_ratios(l, r))
    bold_ratios.append(bold_rat)

fig = plt.figure()

ax = fig.add_subplot(121, projection='3d')
ax.set_title("Candidate Parameter Ratios")

x, y, z = zip(*param_ratios)
p = ax.scatter(x, y, z, c=bold_ratios, cmap='gist_heat')
fig.colorbar(p)

# Add sphere for mean
x = np.mean([r[0] for r in param_ratios])
y = np.mean([r[1] for r in param_ratios])
z = np.mean([r[2] for r in param_ratios])
print("Ratio mean: %9.4f %9.4f %9.4f" % (x,y,z))
ax.scatter([x], [y], [z], s=1000, c='cyan')

# Plot parameter ratios colored by polarized BOLD ratios
param_ratios = []
bold_ratios = []
for l,r,lat_rat,bold_rat in candidates:
    param_ratios.append(polarize_ratios(get_param_ratios(l, r)))
    bold_ratios.append(bold_rat)

ax = fig.add_subplot(122, projection='3d')
ax.set_title("Polarized Ratios")

x, y, z = zip(*param_ratios)
p = ax.scatter(x, y, z, c=bold_ratios, cmap='gist_heat')
fig.colorbar(p)

# Add sphere for mean
x = np.mean([r[0] for r in param_ratios])
y = np.mean([r[1] for r in param_ratios])
z = np.mean([r[2] for r in param_ratios])
print("Ratio mean (polarized): %9.4f %9.4f %9.4f" % (x,y,z))
ax.scatter([x], [y], [z], s=1000, c='cyan')

plt.show()


# Divide ratios by greater than or less than one (8 bins)
quadrants = dict()
for (a,b,c),br in zip(param_ratios, bold_ratios):
    key = []
    for k in a,b,c:
        if k < -0.1:
            key.append(0)
        elif k > 0.1:
            key.append(2)
        else:
            key.append(1)

    try:
        quadrants[tuple(key)].append((a,b,c,br))
    except KeyError:
        quadrants[tuple(key)] = [(a,b,c,br)]

print("\ns/av/va <=>  Count    sens        av        va      BOLD      mean")
for k,v in sorted(quadrants.iteritems(), key = lambda x: len(x[1])):
    a, b, c, br = zip(*v)
    ma, mb, mc, mbr = np.mean(a), np.mean(b), np.mean(c), np.mean(br)
    print(("%s %6d " + "%9.4f " * 5) % (k, len(v), ma, mb, mc, mbr, np.linalg.norm((ma, mb, mc))))


# Plot vectors for candidate pairs
vectors = []
for l,r,lat_rat,bold_rat in candidates:
    vectors.append(get_params(l) + \
        tuple(x - y for x,y in zip(get_params(r), get_params(l))))

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.set_title("Candidate Parameter Vectors")

x, y, z, u, v, w = zip(*vectors)
ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.05)
plt.show()
