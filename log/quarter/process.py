import os
import numpy as np

scale_factor = 0.0

for d in ("healthy", "ptsd"):
    amy = []
    mean = []
    stdev = []

    for f in os.listdir(d):
        if f.endswith(".txt"):
            for line in open("%s/%s" % (d, f)):
                if line.startswith("Using amygdala sensitivity:"):
                    amy.append(float(line.split(":")[1].strip()))
                elif line.startswith("Response latency mean:"):
                    mean.append(float(line.split(":")[1].strip()))
                elif line.startswith("Response latency std dev:"):
                    stdev.append(float(line.split(":")[1].strip()))

    mean_amy = np.mean(amy)
    mean_mean = np.mean(mean)
    mean_stdev = np.mean(stdev)

    stdev_amy = np.std(amy)
    stdev_mean = np.std(mean)
    stdev_stdev = np.std(stdev)

    # ms per timestep
    if scale_factor == 0.0:
        scale_factor = (440.0 - 120.0) / mean_mean

    print("%s condition:" % d)
    print("Amygdala sensitivity:")
    print(mean_amy)
    print(stdev_amy)
    print("")
    print("Mean latency:")
    print(mean_mean, mean_mean * scale_factor, mean_mean * scale_factor + 120)
    print(stdev_mean, stdev_mean * scale_factor)
    print("")
    print("Std dev latency:")
    print(mean_stdev, mean_stdev * scale_factor)
    print(stdev_stdev, stdev_stdev * scale_factor)
    print("")
