import os
import numpy as np
import matplotlib.pyplot as plt

scale_factor = 0.0

for d in ("healthy", "ptsd"):
    amy = []
    mean = []
    stdev = []
    av = []
    va = []
    bold_amy = []
    bold_vmpfc = []
    activ_amy = []
    activ_vmpfc = []

    for f in os.listdir(d):
        if f.endswith(".txt"):
            for line in open("%s/%s" % (d, f)):
                if line.startswith("Using amygdala sensitivity:"):
                    amy.append(float(line.split(":")[1].strip()))
                elif line.startswith("Response latency mean:"):
                    mean.append(float(line.split(":")[1].strip()))
                elif line.startswith("Response latency std dev:"):
                    stdev.append(float(line.split(":")[1].strip()))
                elif line.startswith("Using amygdala->vmPFC:"):
                    av.append(float(line.split(":")[1].strip()))
                elif line.startswith("Using vmPFC->amygdala:"):
                    va.append(float(line.split(":")[1].strip()))
                elif line.startswith("BOLD Amygdala:"):
                    bold_amy.append(float(line.split(":")[1].strip()))
                elif line.startswith("BOLD vmPFC:"):
                    bold_vmpfc.append(float(line.split(":")[1].strip()))
                elif line.startswith("Activation Amygdala:"):
                    activ_amy.append(float(line.split(":")[1].strip()))
                elif line.startswith("Activation vmPFC:"):
                    activ_vmpfc.append(float(line.split(":")[1].strip()))
            print("%8.4f " * 4 % (amy[-1], av[-1], va[-1], mean[-1]))



    mean_amy = np.mean(amy)
    mean_mean = np.mean(mean)
    mean_stdev = np.mean(stdev)
    mean_av = np.mean(av)
    mean_va = np.mean(va)
    mean_bold_amy = np.mean(bold_amy)
    mean_bold_vmpfc = np.mean(bold_vmpfc)
    mean_activ_amy = np.mean(activ_amy)
    mean_activ_vmpfc = np.mean(activ_vmpfc)

    stdev_amy = np.std(amy)
    stdev_mean = np.std(mean)
    stdev_stdev = np.std(stdev)

    # ms per timestep
    if scale_factor == 0.0:
        scale_factor = (440.0 - 120.0) / mean_mean

    print("\n%s condition:" % d)
    print("Mean Amygdala sensitivity:")
    print(mean_amy)
    print("")
    print("Mean amygdala -> vmPFC strength:")
    print(mean_av)
    print("")
    print("Mean vmPFC -> amygdala strength:")
    print(mean_va)
    print("")
    print("Mean latency:")
    print(mean_mean, mean_mean * scale_factor, mean_mean * scale_factor + 120)
    print("")
    print("Std dev latency:")
    print(mean_stdev, mean_stdev * scale_factor)
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

    plt.subplot(211)
    plt.hist(mean)
    plt.subplot(212)
    plt.hist(bold_amy)
    plt.show()
