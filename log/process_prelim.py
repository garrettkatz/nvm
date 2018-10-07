import os
import numpy as np
import matplotlib.pyplot as plt

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

        # Add recurrent activity to BOLD?
        self.bold_amy = bold_amy + ((1.0 - 0.025) * activ_amy)
        self.bold_vmpfc = bold_vmpfc + ((1.0 - 0.00125) * activ_vmpfc)
        #self.bold_amy = bold_amy
        #self.bold_vmpfc = bold_vmpfc

        self.activ_amy = activ_amy
        self.activ_vmpfc = activ_vmpfc
        self.max_activ_amy = max_activ_amy
        self.max_activ_vmpfc = max_activ_vmpfc
        self.num_responses = num_responses
        self.steady = Ra * (s - va * tonic) / (1 + (Ra * Rv * av * va)) 

    def valid(self, num_faces):
        return num_faces == self.num_responses

    def clipped(self, threshold):
        return self.mean_latency <= threshold

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

tonic = 0.0                                                                     
Ra = 1.0                                                                        
Rv = 20.0 

samples = []

d = "prelim"
for f in sorted(os.listdir(d)):
    if f.endswith(".txt"):

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
        num_responses = len(tuple(l for l in open("%s/%s" % (d, f)) if "correct" in l))

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



mean_latency = np.mean([s.mean_latency for s in samples])
stdev_latency = np.mean([s.stdev_latency for s in samples])

mean_bold_amy = np.mean([s.bold_amy for s in samples])
mean_bold_vmpfc = np.mean([s.bold_vmpfc for s in samples])
mean_activ_amy = np.mean([s.activ_amy for s in samples])
mean_activ_vmpfc = np.mean([s.activ_vmpfc for s in samples])

# ms per timestep
scale_factor = (440.0 - 120.0) / mean_latency

print("Mean latency:")
print(mean_latency, mean_latency * scale_factor, mean_latency * scale_factor + 120)
print("")
print("Mean std dev latency:")
print(stdev_latency, stdev_latency * scale_factor)
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
