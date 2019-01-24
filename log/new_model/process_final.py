import os
from math import isnan, log, exp
import numpy as np
import matplotlib as cm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import scipy as sc
from random import choice

class Sample:
    def __init__(self, filename, amy_s, vmpfc_s, av, va, lpfc,
            mean_latency, stdev_latency,
            bold_amy, bold_vmpfc, activ_amy, activ_vmpfc, max_activ_amy,
            max_activ_vmpfc, num_responses, center):
        self.filename = filename
        self.amy_s = amy_s
        self.vmpfc_s = vmpfc_s
        self.av = av
        self.va = va
        self.lpfc = lpfc
        self.mean_latency = mean_latency
        self.stdev_latency = stdev_latency

        self.bold_amy = bold_amy
        self.bold_vmpfc = bold_vmpfc

        self.activ_amy = activ_amy
        self.activ_vmpfc = activ_vmpfc
        self.max_activ_amy = max_activ_amy
        self.max_activ_vmpfc = max_activ_vmpfc
        self.num_responses = num_responses
        self.center = center

    def valid(self, num_faces):
        return num_faces == self.num_responses and self.center == num_faces

    def to_3d(self):
        return (self.amy_s, self.va, self.lpfc)

    def activ_bounded(self, bound):
        return self.max_activ_amy < bound and self.max_activ_vmpfc < bound

    def __str__(self):
        return (" %20s: " + ("%8.4f " * 8)) % (
            self.filename, self.amy_s, self.va, self.lpfc, self.mean_latency,
            self.bold_amy, self.bold_vmpfc, self.max_activ_amy, self.max_activ_vmpfc)

    def __eq__(self, other): 
        return self.amy_s == other.amy_s \
            and self.vmpfc_s == other.vmpfc_s \
            and self.av == other.av \
            and self.va == other.va \
            and self.lpfc == other.lpfc \
            and self.mean_latency == other.mean_latency \
            and self.stdev_latency == other.stdev_latency \
            and self.bold_amy == other.bold_amy \
            and self.bold_vmpfc == other.bold_vmpfc \
            and self.activ_amy == other.activ_amy \
            and self.activ_vmpfc == other.activ_vmpfc \
            and self.max_activ_amy == other.max_activ_amy \
            and self.max_activ_vmpfc == other.max_activ_vmpfc \
            and self.num_responses == other.num_responses

directories = ["final/healthy", "final/ptsd", "final/recovered"]

data = dict()
for d in directories:
    data[d] = []
    count = 1
    for f in sorted(os.listdir(d)):
        if f.endswith(".txt"):

            amy_s = 0.0
            vmpfc_s = 0.0
            av = 0.0
            va = 0.0
            lpfc = 0.0
            mean_latency = 0.0
            stdev_latency = 0.0
            bold_amy = 0.0
            bold_vmpfc = 0.0
            activ_amy = 0.0
            activ_vmpfc = 0.0
            max_activ_amy = 0.0
            max_activ_vmpfc = 0.0
            num_responses = len(tuple(l for l in open("%s/%s" % (d, f)) if "correct" in l))
            center = 0

            for line in open("%s/%s" % (d, f)):
                if line.startswith("Using amygdala sensitivity:"):
                    amy_s = (float(line.split(":")[1].strip()))
                elif line.startswith("Using vmPFC sensitivity:"):
                    vmpfc_s = (float(line.split(":")[1].strip()))
                elif line.startswith("Using amygdala->vmPFC:"):
                    av = (float(line.split(":")[1].strip()))
                elif line.startswith("Using vmPFC->amygdala:"):
                    va = (float(line.split(":")[1].strip()))
                elif line.startswith("Using lPFC->vmPFC:"):
                    lpfc = (float(line.split(":")[1].strip()))
                elif line.startswith("Response latency mean:"):
                    mean_latency = (float(line.split(":")[1].strip()))
                elif line.startswith("Response latency std dev:"):
                    stdev_latency = (float(line.split(":")[1].strip()))
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
                elif "center" in line:
                    center += 1

            if mean_latency != 0.0:
                count += 1
                data[d].append(Sample(f, amy_s, vmpfc_s, av, va, lpfc,
                    mean_latency, stdev_latency,
                    bold_amy, bold_vmpfc, activ_amy, activ_vmpfc, max_activ_amy,
                    max_activ_vmpfc, num_responses, center))

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Plot latency of good data points using heatmap
colors = ('blue', 'red', 'green')
for i,key in enumerate(directories):
    samples = data[key]

    # Filter samples with amy_s == 0 or va == 0
    print(key)
    print("amy_s:      %12.4f %12.4f" % (
        np.mean([s.amy_s for s in samples]),
        np.std([s.amy_s for s in samples])))
    print("va:         %12.4f %12.4f" % (
        np.mean([s.va for s in samples]),
        np.std([s.va for s in samples])))
    print("lpfc:       %12.4f %12.4f" % (
        np.mean([s.lpfc for s in samples]),
        np.std([s.lpfc for s in samples])))
    print("Latency:    %12.4f %12.4f" % (
        np.mean([s.mean_latency for s in samples]),
        np.std([s.mean_latency for s in samples])))
    print("BOLD amy:   %12.4f %12.4f" % (
        np.mean([s.bold_amy for s in samples]),
        np.std([s.bold_amy for s in samples])))
    print("BOLD vmpfc: %12.4f %12.4f" % (
        np.mean([s.bold_vmpfc for s in samples]),
        np.std([s.bold_vmpfc for s in samples])))

    x, y, z = zip(*[(s.amy_s, s.va, s.lpfc) for s in samples])
    ax1.scatter(x, y, z, c=colors[i])
    x, y, z = zip(*[(s.amy_s, s.va, s.mean_latency) for s in samples])
    #c = [s.lpfc for s in samples]
    c = colors[i]
    ax2.scatter(x, y, z, c=c, cmap='gist_heat')

ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 2.0])
ax1.set_zlim([1.0, 2.0])

ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 2.0])
ax2.set_zlim([100, 500])

ax1.set_xlabel("amy_s")
ax1.set_ylabel("va")
ax1.set_zlabel("lpfc")

ax2.set_xlabel("amy_s")
ax2.set_ylabel("va")
ax2.set_zlabel("latency")

try:
    plt.show()
except: pass

plt.subplot(111)
plt.title("Latency Histogram")
points = []
for d in directories:
    points.append(tuple(s.mean_latency for s in data[d]))
plt.hist(points, 25, histtype='bar', stacked=True)
plt.legend([d.split("/")[-1] for d in directories])
plt.xlabel("Latency")
plt.ylabel("Count")

try:
    plt.show()
except: pass
