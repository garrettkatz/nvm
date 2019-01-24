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
from random import sample, shuffle

### PARAMETERS ###

# Number of faces in the trial
num_faces = 5

# Latency range to consider
lat_min = 61.2
lat_max = 1000

# Tolerances for significant changes (percentage of original)
lat_tolerance = 0.05
lat_tol_min = 1.0 - lat_tolerance
lat_tol_max = 1.0 + lat_tolerance

bold_tolerance = 0.0
bold_tol_min = 1.0 - bold_tolerance
bold_tol_max = 1.0 + bold_tolerance

param_tolerance = 0.0
param_tol_min = 1.0 - param_tolerance
param_tol_max = 1.0 + param_tolerance


# Bound on biologically realistic parameter changes
bound = 0.25
bound_min = 1.0 - bound
bound_max = 1.0 + bound

# Entropy reliability cutoff
ent_max = 0.3

# Target latency
lat_target = 633.0 / 440
lat_target_min = lat_tol_min * lat_target
lat_target_max = lat_tol_max * lat_target

targets = [lat_target, 1.05, 0.95]
log_targets = [log(x) for x in targets]

##################


def maybe_plot():
    try:
        plt.show()
        #plt.clear()
    except: pass

def best_fit_slope_and_intercept(xs,ys):
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    m = (((mean_x * mean_y) - np.mean(xs*ys)) /
         ((mean_x * mean_x) - np.mean(xs*xs)))
    b = mean_y - m*mean_x
    return m, b


class Sample:
    def __init__(self, filename, amy_s, vmpfc_s, av, va, lpfc,
            mean_latency, stdev_latency,
            bold_amy, bold_vmpfc, activ_amy, activ_vmpfc, max_activ_amy,
            max_activ_vmpfc, num_responses, center, iterations):
        self.filename = filename
        self.amy_s = amy_s
        self.vmpfc_s = vmpfc_s
        self.av = av
        self.va = va
        self.lpfc = lpfc
        self.mean_latency = mean_latency
        self.stdev_latency = stdev_latency

        self.bold_amy = bold_amy / iterations
        self.bold_vmpfc = bold_vmpfc / iterations

        self.activ_amy = activ_amy / iterations
        self.activ_vmpfc = activ_vmpfc / iterations
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
        return (" %20s: " + ("%10.2f " * 8)) % (
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

    def __hash__(self):
        return hash(self.to_3d())

class Pair:
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2
        self.amy_s_ratio = s2.amy_s / s1.amy_s
        self.va_ratio = s2.va / s1.va
        self.lpfc_ratio = s2.lpfc / s1.lpfc
        self.mean_latency_ratio = s2.mean_latency / s1.mean_latency
        self.bold_amy_ratio = s2.bold_amy / s1.bold_amy
        self.bold_vmpfc_ratio = s2.bold_vmpfc / s1.bold_vmpfc
        self.ratios =  (
            self.amy_s_ratio,
            self.va_ratio,
            self.lpfc_ratio,
            self.mean_latency_ratio,
            self.bold_amy_ratio,
            self.bold_vmpfc_ratio)
        self.log_ratios = [log(r) for r in self.ratios]
        self.deltas = (
            s2.amy_s - s1.amy_s,
            s2.va - s1.va,
            s2.lpfc - s1.lpfc)
        self.changes = tuple(-1 if ratio < param_tol_min else
            (1 if ratio > param_tol_max else 0)
                for ratio in self.ratios[:3])
        self.key = "| %s |" % " ".join(tuple('-' if c == -1 else (
            '+' if c == 1 else ' ') for c in self.changes))

        # Target latency
        self.check_latency = self.ratios[3] < lat_target_max and self.ratios[3] > lat_target_min

        # Amygdala BOLD increase, vmPFC BOLD decrease
        self.check_bold = self.ratios[4] > bold_tol_max and self.ratios[5] < bold_tol_min

        # Amygdala BOLD increase, vmPFC BOLD decrease
        self.check_wm = self.va_ratio < 1.0

class Slice:
    def __init__(self, params, samples):
        self.params = params
        self.samples = [s for s in samples
            if all(p is None or p == sp for (p,sp) in zip(params, s.to_3d()))]

        self.index = params.index(None)
        self.x_vals = np.array([s.to_3d()[self.index] for s in self.samples])
        self.y_vals = np.array([s.mean_latency for s in self.samples])

        if len(self.samples) > 1:
            self.fluc = float(max(self.y_vals)) / min(self.y_vals)
            self.m,self.b = best_fit_slope_and_intercept(self.x_vals, self.y_vals)
        else:
            self.fluc = 0
            self.m = 0
            self.b = 0

    def __str__(self):
        return str(self.params)

class ChangeSet:
    def __init__(self, change, pairs):
        self.change = change
        self.pairs = tuple(p for p in pairs if p.changes == change)
        self.key = self.pairs[0].key

        self.increased = 0
        self.decreased = 0
        self.same = 0

        for p in self.pairs:
            #if p.mean_latency_ratio > lat_tol_max: self.increased += 1
            #elif p.mean_latency_ratio < lat_tol_min: self.decreased += 1
            #else: self.same += 1
            if p.mean_latency_ratio > 1.0: self.increased += 1
            else: self.decreased += 1

        self.fracs = tuple(float(x) / len(self.pairs)
            for x in (self.increased,self.decreased,self.same))
        self.entropy_2 = sc.stats.entropy(self.fracs[:2], base=2)
        self.entropy_3 = sc.stats.entropy(self.fracs, base=3)
        self.entropy = self.entropy_2
        self.max_i = self.fracs.index(max(self.fracs))

    def __str__(self):
        return "%20s:   %8d   %5f   %5f   %5f     %5f" % (
            self.key,
            len(self.pairs),
            self.fracs[0], self.fracs[1], self.fracs[2],
            self.entropy)

def load_data(d, number_faces, latency_minimum, latency_maximum):
    samples = []
    timeout = 0
    baseline = 0

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
            iterations = 0

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
                elif "Time averaged over" in line:
                    iterations = int(line.split()[3])

            if not (center == num_responses == number_faces):
                timeout += 1
            elif mean_latency <= latency_minimum:
                baseline += 1

            # Filter samples with failed trials, amy_s == 0, or va == 0
            if mean_latency > latency_minimum and mean_latency < latency_maximum \
                and center == number_faces \
                and center == num_responses \
                and amy_s > 0.0 and va > 0.0:
                samples.append(Sample(f, amy_s, vmpfc_s, av, va, lpfc,
                    mean_latency, stdev_latency,
                    bold_amy, bold_vmpfc, activ_amy, activ_vmpfc, max_activ_amy,
                    max_activ_vmpfc, num_responses, center, iterations))

    return samples, baseline, timeout

def split_samples(samples, test_ratio):
    samples_copy = [s for s in samples]
    shuffle(samples_copy)
    split_index = int(len(samples_copy) * test_ratio)
    return samples_copy[split_index:], samples_copy[:split_index]

def hist_latency(samples):
    # Plot latencies and BOLD
    plt.title("Mean Antisaccade Latency")
    plt.hist(tuple(s.mean_latency for s in samples), 50)
    plt.xlabel("Mean Latency")
    plt.ylabel("Count")

    maybe_plot()

def plot_space(samples):
    fig = plt.figure()

    # Plot latency of good data points using heatmap
    ax = fig.add_subplot(131, projection='3d')
    ax.set_title("lPFC -> vmPFC")

    data = np.array([(s.amy_s, s.va, s.mean_latency) for s in samples])
    labels = np.array([s.lpfc for s in samples])
    x, y, z = zip(*data)
    p = ax.scatter(x, y, z, c=labels, cmap='gist_heat')
    fig.colorbar(p)

    ax.set_xlabel("amy_s")
    ax.set_ylabel("va")
    ax.set_zlabel("latency")
    ax.set_zlim([lat_min, lat_max])

    ax = fig.add_subplot(132, projection='3d')
    ax.set_title("vmPFC -> amygdala")

    data = np.array([(s.lpfc, s.amy_s, s.mean_latency) for s in samples])
    labels = np.array([s.va for s in samples])
    x, y, z = zip(*data)
    p = ax.scatter(x, y, z, c=labels, cmap='gist_heat')
    fig.colorbar(p)

    ax.set_xlabel("lpfc")
    ax.set_ylabel("amy_s")
    ax.set_zlabel("latency")
    ax.set_zlim([lat_min, lat_max])

    ax = fig.add_subplot(133, projection='3d')
    ax.set_title("temporal -> amygdala")

    data = np.array([(s.va, s.lpfc, s.mean_latency) for s in samples])
    labels = np.array([s.amy_s for s in samples])
    x, y, z = zip(*data)
    p = ax.scatter(x, y, z, c=labels, cmap='gist_heat')
    fig.colorbar(p)

    ax.set_xlabel("va")
    ax.set_ylabel("lpfc")
    ax.set_zlabel("latency")
    ax.set_zlim([lat_min, lat_max])

    maybe_plot()


    fig = plt.figure()

    # Plot latency of good data points using heatmap
    ax = fig.add_subplot(131, projection='3d')
    ax.set_title("Mean Antisaccade Latency")

    data = np.array([s.to_3d() for s in samples])
    labels = np.array([s.mean_latency for s in samples])
    x, y, z = zip(*data)
    p = ax.scatter(x, y, z, c=labels, cmap='gist_heat', norm=cm.colors.LogNorm())
    fig.colorbar(p, ticks=[100, 200, 300, 400, 500, 600, 700, 800], format=cm.ticker.LogFormatter(10, labelOnlyBase=False, minor_thresholds=(10, 10)))

    ax.set_xlabel("at")
    ax.set_ylabel("av")
    ax.set_zlabel("vl")

    ax = fig.add_subplot(132, projection='3d')
    ax.set_title("Amygdala BOLD")

    data = np.array([s.to_3d() for s in samples])
    labels = np.array([s.bold_amy for s in samples])
    x, y, z = zip(*data)
    p = ax.scatter(x, y, z, c=labels, cmap='gist_heat', norm=cm.colors.LogNorm())
    fig.colorbar(p, format=cm.ticker.LogFormatter(10, labelOnlyBase=False))

    ax.set_xlabel("at")
    ax.set_ylabel("av")
    ax.set_zlabel("vl")

    ax = fig.add_subplot(133, projection='3d')
    ax.set_title("vmPFC BOLD")

    data = np.array([s.to_3d() for s in samples])
    labels = np.array([s.bold_vmpfc for s in samples])
    x, y, z = zip(*data)
    p = ax.scatter(x, y, z, c=labels, cmap='gist_heat', norm=cm.colors.LogNorm())
    fig.colorbar(p, format=cm.ticker.LogFormatter(10, labelOnlyBase=False))

    ax.set_xlabel("at")
    ax.set_ylabel("av")
    ax.set_zlabel("vl")

    maybe_plot()



# Perform linear regression to predict latencies in good samples
def linear_regression(samples):
    data = [s.to_3d() for s in samples]
    labels = [s.mean_latency for s in samples]
    model = LinearRegression()
    model.fit(data, labels)
    print("%30s : %f %s %f" % ("Latency model",
        model.score(data, labels), model.coef_, model.intercept_))

    data = [s.to_3d() for s in samples]
    labels = [s.bold_amy for s in samples]
    model = LinearRegression()
    model.fit(data, labels)
    print("%30s : %f %s %f" % ("BOLD amygdala model",
        model.score(data, labels), model.coef_, model.intercept_))

    data = [s.to_3d() for s in samples]
    labels = [s.bold_vmpfc for s in samples]
    model = LinearRegression()
    model.fit(data, labels)
    print("%30s : %f %s %f" % ("BOLD vmPFC model",
        model.score(data, labels), model.coef_, model.intercept_))

    data = [s.to_3d() for s in samples]
    labels = [s.max_activ_vmpfc for s in samples]
    model = LinearRegression()
    model.fit(data, labels)
    print("%30s : %f %s %f" % ("Max activ vmPFC model",
        model.score(data, labels), model.coef_, model.intercept_))
    print("")


def create_candidates(samples):
    candidates = []
    organized_candidates = dict()

    for s in samples:
        cs = [Pair(s, s2) for s2 in samples if s != s2]
        organized_candidates[s] = cs
        candidates += cs

    return candidates, organized_candidates

def build_model(samples):
    # Create candidate parameter set pairs
    candidates, organized_candidates = create_candidates(samples)

    # Coefficients mapping parameter changes to latency/BOLD changes
    coefficients = []

    print("Model fits:")
    for i,label in enumerate(["Latency", "Amygdala BOLD", "vmPFC BOLD"]):
        data = [c.log_ratios[:3] for c in candidates]
        labels = [c.log_ratios[3+i] for c in candidates]

        model = LinearRegression(fit_intercept=False)
        model.fit(data, labels)
        coefficients.append(model.coef_)

        score = model.score(data, labels)
        print("%30s : %f %s %f" % (
            ("%s model" % label),
            score,
            model.coef_,
            model.intercept_))
    print

    # Solve for target values to create a suggested change vector
    a = np.array(coefficients)
    sugg = np.linalg.solve(a, log_targets)
    print("Suggested change: %s" % np.array(sugg))
    print

    return coefficients, sugg

def validate(samples, coefficients, sugg):
    candidates, organized_candidates = create_candidates(samples)

    results = []
    correct = np.array([0, 0, 0])
    for c in candidates:
        predicted = np.dot(coefficients, c.log_ratios[:3])
        actual = c.log_ratios[3:]
        dist = sc.spatial.distance.cosine(sugg, c.log_ratios[:3])
        size = np.linalg.norm(c.log_ratios[:3])

        eq = np.equal(np.sign(actual), np.sign(predicted))
        results.append((c, actual, predicted, eq, dist, size))
        correct = np.add(correct, eq)

    data = tuple(d for c,a,p,e,d,s in results)
    counts,bins = np.histogram(data, 50)

    data = tuple(d for c,a,p,e,d,s in results
        if a[0] > 0 and a[1] > 0 and a[2] < 0)
    good = np.histogram(data, bins)[0]

    percentages = [float(g) / c for g,c in zip(good,counts)]
    plt.plot(bins[:-1], percentages)

    plt.title("Probability of Latency Increase")
    plt.xlabel("Cosine Distance from Suggested")
    plt.ylabel("Probability")
    maybe_plot()
    



    accuracy = tuple(float(x) / len(candidates) for x in correct)

    print("Validation accuracy:")
    for i,label in enumerate(["Latency", "Amygdala BOLD", "vmPFC BOLD"]):
        print("%30s : %f" % (label, accuracy[i]))
    print

    return accuracy

def search(samples, coeff_means, sugg):
    candidates, organized_candidates = create_candidates(samples)

    bests = []
    for s in samples:
        distances = []
        cs = organized_candidates[s]
        for c in cs:
            #dist = sc.spatial.distance.cosine(sugg, c.log_ratios[:3])
            dist = sc.spatial.distance.cosine(log_targets, c.log_ratios[3:])
            distances.append((c,dist))
        best = min(distances, key=lambda x:x[1])
        bests.append(best)

    c,d = min(bests, key=lambda x: x[1])
    print(c.s1)
    print(c.s2)
    print(("%10.4f " * 6) % c.ratios)
    print(d)
    print(sc.spatial.distance.cosine(sugg, c.log_ratios[:3]))
    print

    params = c.s1.to_3d()
    dest = [exp(sugg[i]) * params[i] for i in range(3)]
    source_stds = [(exp(log(1 + (124./440)) / coeff_means[0][i])) for i in range(3)]
    dest_stds =   [(exp(log(1 + (195./633)) / coeff_means[0][i])) for i in range(3)]
    print("Source", params)
    print("Dest", dest)
    print("Source Stdev", source_stds)
    print("Dest Stdev", dest_stds)
    print("Source Stdev", [(params[i] * source_stds[i]) - params[i] for i in range(3)])
    print("Dest Stdev", [(dest[i] * dest_stds[i] - dest[i]) for i in range(3)])

    predictions = []
    while len(predictions) < 1000:
        sample = []
        for i in range(3):
            sample.append(np.random.normal(params[i], 0.2 * params[i]))
        try:
            c_logs = [log(sample[i] / params[i]) for i in range(3)]
            predictions.append(exp(np.dot(coeff_means[0], c_logs)))
        except: pass

    print(np.mean(predictions))
    print(np.std(predictions))

    fig = plt.figure()
    plt.title("Latency predictions")
    plt.hist(predictions, 100)
    plt.xlabel("Latency")
    plt.ylabel("Count")
    maybe_plot()

    predictions = []
    while len(predictions) < 1000:
        sample = []
        for i in range(3):
            sample.append(np.random.normal(dest[i], 0.2 * dest[i]))
        try:
            c_logs = [log(sample[i] / dest[i]) for i in range(3)]
            predictions.append(exp(np.dot(coeff_means[0], c_logs)))
        except: pass

    print(np.mean(predictions))
    print(np.std(predictions))

    fig = plt.figure()
    plt.title("Latency predictions")
    plt.hist(predictions, 100)
    plt.xlabel("Latency")
    plt.ylabel("Count")
    maybe_plot()





print("Loading samples...")
samples, baseline, timeout = load_data("explore", num_faces, lat_min, lat_max)
#samples = split_samples(samples, 0.25)[1]
print("Valid unclipped samples: %d" % len(samples))
print("       Baseline samples: %d" % baseline)
print("      Timed out samples: %d" % timeout)
print("")

#training,test = split_samples(samples, 0.3)
#print("Training: %d" % len(training))
#print("Test: %d" % len(test))
#print("")

#hist_latency(samples)
plot_space(samples)
#linear_regression(samples)

#coeff, sugg = build_model(samples)
#validate(samples, coeff, sugg)

#search(samples, coeff, sugg)





candidates, organized_candidates = create_candidates(samples)
#foo



# Create sets according to direction of change
print("Examining change sets...")
changes = set(c.changes for c in candidates)
change_sets = sorted(
    tuple(ChangeSet(change, candidates) for change in changes),
    key = lambda x: x.entropy)

format_string = "%20s:   %8d   %5f   %5f   %5f     %5f"
header = "                 Key       Total   Increase   Decrease   Same         Entropy "

key_names = ("at", "av", "vl")
key_entropies = dict()

for cs in change_sets:
    key_entropies[cs.key] = cs.entropy

for i,label in enumerate(("Increase", "Decrease", "Same")):
    print("%s latency" % label)
    for cs in change_sets:
        if cs.max_i == i:
            print(cs)
    print("")

print("Reliably increases latency")
for cs in change_sets:
    if cs.max_i == 0 and cs.entropy < ent_max:
        print(cs)
print("")



# Slice analysis
slice_points = [set() for i in range(3)]
for s in samples:
    params = s.to_3d()
    for index in range(len(params)):
        slice_points[index].add(tuple(None if i == index else p for i,p in enumerate(params)))

slices = dict()
for i,k in enumerate(key_names):
    slices[k] = tuple(Slice(p, samples) for p in slice_points[i])
    slices[k] = tuple(s for s in slices[k] if len(s.samples) > 0 and s.fluc > lat_tol_max)


fig = plt.figure()
for index,main_key in enumerate(key_names):
    ax = fig.add_subplot(131 + index)
    ax.set_xlabel(main_key)
    ax.set_ylabel("Mean Latency")

    inc,dec = 0,0

    for sl in slices[main_key]:
        plt.plot(sl.x_vals, sl.y_vals)
        if sl.m > 0:
            inc +=1
        else:
            dec += 1

    print("%20s: %5d+ %5d-" % (main_key, inc,dec))
print("")

maybe_plot()




all_bins = dict()
filtered_bins = dict()
final_candidates = []
for c in candidates:
    if c.check_latency and c.check_bold:
        filtered_bins.setdefault(c.key, []).append((c.s1,c.s2))
        final_candidates.append(c)
    # Require latency
    if c.check_latency:
        all_bins.setdefault(c.key, []).append((c.s1,c.s2))

print("Latency")
pairs = sorted([(k, len(all_bins[k])) for k in all_bins], key = lambda x:key_entropies[x[0]])
for k,v in pairs:
    print("%20s: %8d    ent=%9.4f" % (k, v, key_entropies[k]))
print("")

print("Latency + BOLD")
pairs = sorted([(k, len(filtered_bins[k])) for k in filtered_bins], key = lambda x:key_entropies[x[0]])
for k,v in pairs:
    print("%20s: %8d    ent=%9.4f" % (k, v, key_entropies[k]))
print("")


print("\n\nRandomly selected candidates:")
try:
    # Select a candidate where va decreases (decreased white matter)
    selected = sample([c for c in final_candidates if c.check_wm], 10)
    for c in selected:
        print(c.s1)
        print(c.s2)
        print(("%10.4f " * 6) % c.ratios)
        print("")
except: pass

'''
foobar = [c for c in final_candidates if c.key == "|   - - |"]
for c in foobar:
    print(c.s1)
    print(c.s2)
    print(("%10.4f " * 6) % c.ratios)
    print("")
'''






################################################################################
'''


# Prepare vectors for candidate healthy->PTSD transformations
def get_vectors(ratios=False):
    out = []
    keys = set(c.key for c in final_candidates)
    sizes = dict((k, len(tuple(c for c in final_candidates if c.key == k))) for k in keys)
    keys = sorted(keys, key = lambda k: sizes[k], reverse=True)
    for k in keys:
        if ratios:
            from_points = tuple((0,0,0) for c in final_candidates if c.key==k)
            to_points = tuple(tuple(log(r) for r in ratios[:3])
                for c in final_candidates if c.key==k)
        else:
            from_points = tuple(c.s1.to_3d() for c in final_candidates if c.key==k)
            to_points = tuple(c.s2.to_3d() for c in final_candidates if c.key==k)
        x,y,z = np.array(zip(*from_points))
        u,v,w = np.array(zip(*to_points))
        out.append((x,y,z,u-x,v-y,w-z))
    return out,keys

fig = plt.figure()

# Plot latency of good data points using heatmap
ax = fig.add_subplot(221, projection='3d')
ax.set_title("BOLD amygdala")

data = np.array([s.to_3d() for s in unclipped])
labels = np.array([s.bold_amy for s in unclipped])
x, y, z = zip(*data)
p = ax.scatter(x, y, z, c=labels, cmap='gist_heat')
fig.colorbar(p)

ax.set_xlabel("amy_s")
ax.set_ylabel("va")
ax.set_zlabel("lpfc")

ax = fig.add_subplot(222, projection='3d')
ax.set_title("BOLD vmPFC")

data = np.array([s.to_3d() for s in unclipped])
labels = np.array([s.bold_vmpfc for s in unclipped])
x, y, z = zip(*data)
p = ax.scatter(x, y, z, c=labels, cmap='gist_heat')
fig.colorbar(p)

ax.set_xlabel("amy_s")
ax.set_ylabel("va")
ax.set_zlabel("lpfc")

ax = fig.add_subplot(223, projection='3d')
ax.set_title("Max activation amygdala")

data = np.array([s.to_3d() for s in unclipped])
labels = np.array([s.max_activ_amy for s in unclipped])
x, y, z = zip(*data)
p = ax.scatter(x, y, z, c=labels, cmap='gist_heat')
fig.colorbar(p)

ax.set_xlabel("amy_s")
ax.set_ylabel("va")
ax.set_zlabel("lpfc")

ax = fig.add_subplot(224, projection='3d')
ax.set_title("Max activation vmPFC")

data = np.array([s.to_3d() for s in unclipped])
labels = np.array([s.max_activ_vmpfc for s in unclipped])
x, y, z = zip(*data)
p = ax.scatter(x, y, z, c=labels, cmap='gist_heat')
fig.colorbar(p)

ax.set_xlabel("amy_s")
ax.set_ylabel("va")
ax.set_zlabel("lpfc")

maybe_plot()


fig = plt.figure()

ax = fig.add_subplot(121, projection='3d')
ax.set_title("Candidate Transformations")

colors = ('blue', 'green', 'red', 'orange', 'purple', 'yellow')
vectors,keys = get_vectors(False)
for i,(x,y,z,u,v,w) in enumerate(vectors):
    ax.quiver(x,y,z,u,v,w, length=0.1, normalize=True, color=colors[i%len(colors)])
ax.legend(keys)

x,y,z = zip(*tuple(s.to_3d() for s in unclipped))
ax.set_xlim([min(x), max(x)])
ax.set_ylim([min(y), max(y)])
ax.set_zlim([min(z), max(z)])
ax.set_xlabel("amy_s")
ax.set_ylabel("va")
ax.set_zlabel("lpfc")

ax = fig.add_subplot(122, projection='3d')
ax.set_title("Candidate Transformations")

colors = ('blue', 'green', 'red', 'orange', 'purple', 'yellow')
vectors,keys = get_vectors(True)
for i,(x,y,z,u,v,w) in enumerate(vectors):
    ax.quiver(x,y,z,u,v,w, length=0.1, normalize=True, color=colors[i%len(colors)])
ax.legend(keys)

ax.set_xlim([-.1,.1])
ax.set_ylim([-.1,.1])
ax.set_zlim([-.1,.1])
ax.set_xlabel("amy_s")
ax.set_ylabel("va")
ax.set_zlabel("lpfc")

maybe_plot()

'''

# 3D Mesh Plots
'''
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X, Y, Z = (np.array(l) for l in zip(*[s.to_3d() for s in invalid]))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap="cool_warm",
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()




def build_model(samples):
    # Create candidate parameter set pairs
    candidates, organized_candidates = create_candidates(samples)

    # Coefficients mapping parameter changes to latency/BOLD changes
    coefficients = [[] for i in range(3)]

    # Suggested direction of change for target latency/BOLD changes
    suggestions = []

    for s in samples:
        cs = organized_candidates[s]

        # Perform linear regression for each target
        matrix = []
        for i,label in enumerate(["Latency", "Amygdala BOLD", "vmPFC BOLD"]):
            model = LinearRegression(fit_intercept=False)

            data = [c.log_ratios[:3] for c in cs]
            labels = [c.log_ratios[3+i] for c in cs]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Log Space")

            to_plot = np.array(data)
            to_plot_labels = np.array([c.log_ratios[3] for c in cs])
            x, y, z = zip(*to_plot)
            p = ax.scatter(x, y, z, c=to_plot_labels, cmap='gist_heat')
            fig.colorbar(p)

            ax.set_xlabel("amy_s")
            ax.set_ylabel("va")
            ax.set_zlabel("lpfc")
            maybe_plot()

            model.fit(data, labels)

            matrix.append(model.coef_)

            #score = model.score(data, labels)
            #print("%30s %30s : %f %s %f" % (
            #    s.to_3d(),
            #    ("%s model" % label),
            #    score,
            #    model.coef_,
            #    model.intercept_))

        # Collect coefficients
        for i in range(3):
            coefficients[i].append(tuple(x for x in matrix[i]))

        # Solve for target values to create a suggested change vector
        a = np.array(matrix)
        sugg = np.linalg.solve(a, log_targets)
        suggestions.append((s, sugg))

        #print(t)
        #print(estimate)
        #print([exp(e) for e in estimate])
        #print(sugg)
        #print(tuple(exp(e) * p for e,p in zip(estimate, s.to_3d())))
        #print("")

    # Determine directions of change
    directions = dict()
    for s,sugg in suggestions:
        changes = tuple(-1 if r < 0.0 else (1 if r > 0.0 else 0) for r in sugg)
        directions.setdefault(changes, []).append((s,sugg))

    print("Suggested directions of change:")
    for k,v in directions.iteritems():
        print(k, len(v))

    # Compute means of suggestions and coefficients
    sugg_mean = []
    coeff_means = [[] for i in range(3)]
    fig = plt.figure()
    for i in range(3):
        ax = fig.add_subplot(131 + i)
        data = [sugg[i] for s,sugg in suggestions]
        sugg_mean.append(np.mean(data))

        for j in range(3):
            coeff_means[j].append(
                np.mean([coeff[i] for coeff in coefficients[j]]))

        plt.title("Log Change Histogram %d" % i)
        plt.hist(data, 100)
        plt.xlabel("Log Change")
        plt.ylabel("Count")

    print("Mean suggestion: %s" % sugg_mean)
    print("                 %s" % [exp(x) for x in sugg_mean])
    print("Mean coefficients:")
    labels = ["Latency", "Amybdala BOLD", "vmPFC BOLD"]
    for i,cm in enumerate(coeff_means):
        print("  %30s: %s" % (labels[i], cm))
        print("  %30s: %s" % ("", [exp(x) for x in cm]))
    print("")
    maybe_plot()


    # Plot suggestions
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Change Suggestions")

    data = np.array([sugg for s,sugg in suggestions])
    x, y, z = zip(*data)
    ax.scatter(x, y, z)

    ax.set_xlabel("amy_s")
    ax.set_ylabel("va")
    ax.set_zlabel("lpfc")

    maybe_plot()

    return coeff_means, sugg_mean

def validate(samples, coeff_means, sugg_mean):
    candidates, organized_candidates = create_candidates(samples)

    # Validate from real data
    errors = [[] for i in range(3)]
    sign_errors = [0 for i in range(3)]
    sign_error_points = []
    for s1,cs in organized_candidates.iteritems():
        s1_data = (s1.mean_latency, s1.bold_amy, s1.bold_vmpfc)

        sec = 0

        for c in cs:
            s2_data = (c.s2.mean_latency, c.s2.bold_amy, c.s2.bold_vmpfc)
            c_logs = c.log_ratios[:3]

            for i in range(3):
                #predicted = exp(np.dot(coeff_means[i], c.log_ratios[:3]))
                #actual = c.ratios[3+i]
                #err = predicted - actual
                #errors[i].append(err)
                #sign_err = ((predicted > 1.0) != (actual > 1.0))
                #sign_errors[i] += sign_err

                original = s1_data[i]
                predicted = exp(np.dot(coeff_means[i], c_logs)) * s1_data[i]
                actual = s2_data[i]

                if i == 0:
                    scale = 440. / original
                    original = 440.
                    predicted *= scale
                    actual *= scale

                err = predicted - actual
                sign_err = (predicted - original > 1.0) != (actual - original > 1.0)

                errors[i].append(err)
                sign_errors[i] += sign_err

                if i == 0 and sign_err:
                    sign_error_points.append((c, predicted, actual))
                    sec += 1

        #print("%s %7.4f" % (str(s1), float(sec) / len(cs)))

    means = [
        440., #np.mean([s.mean_latency for s in samples]),
        np.mean([s.bold_amy for s in samples]),
        np.mean([s.bold_vmpfc for s in samples])]

    for i,label in enumerate(["Latency", "Amybdala BOLD", "vmPFC BOLD"]):
        err_mean = np.mean(errors[i])
        err_std = np.std(errors[i])
        print("%30s: mean: %10.2f | error mean: %10.7f | error std: %10.8f | errstd/mean: %10.8f | sign errors: %7.5f" % (
            label,
            means[i],
            err_mean,
            err_std,
            err_std / means[i],
            float(sign_errors[i]) / len(candidates)))

    fig = plt.figure()
    for i in range(3):
        ax = fig.add_subplot(131 + i)
        plt.title("Prediction Errors")
        plt.hist(errors[i], 100)
        plt.xlabel("Error")
        plt.ylabel("Count")

    maybe_plot()


    fig = plt.figure()
    data = [actual for c,p,actual in sign_error_points]
    plt.title("Sign Error Actuals")
    plt.hist(data, 100)
    plt.xlabel("Actual")
    plt.ylabel("Count")
    maybe_plot()


    fig = plt.figure()
    plt.scatter([x[1] for x in sign_error_points], [x[2] for x in sign_error_points])
    plt.xlabel("predicted")
    plt.ylabel("actual")
    maybe_plot()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data = list(x[0].log_ratios[:3] for x in sign_error_points)
    x, y, z = zip(*data)
    ax.scatter(x, y, z)
    ax.scatter(sugg_mean[0]*10, sugg_mean[1]*10, sugg_mean[2]*10, c='red')
    maybe_plot()


def search(samples, coeffs, sugg):
    candidates, organized_candidates = create_candidates(samples)

    bests = []
    for s in samples:
        distances = []
        cs = organized_candidates[s]
        for c in cs:
            #dist = sc.spatial.distance.cosine(sugg, c.log_ratios[:3])
            dist = sc.spatial.distance.cosine(log_targets, c.log_ratios[3:])
            distances.append((c,dist))
        best = min(distances, key=lambda x:x[1])
        bests.append(best)

    c,d = min(bests, key=lambda x: x[1])
    print(c.s1)
    print(c.s2)
    print(("%10.4f " * 6) % c.ratios)
    print(d)
    print(sc.spatial.distance.cosine(sugg, c.log_ratios[:3]))
    print

    params = c.s1.to_3d()
    dest = [exp(sugg[i]) * params[i] for i in range(3)]
    source_stds = [(exp(log(1 + (124./440)) / coeffs[0][i])) for i in range(3)]
    dest_stds =   [(exp(log(1 + (195./633)) / coeffs[0][i])) for i in range(3)]
    print("Source", params)
    print("Dest", dest)
    print("Source Stdev", source_stds)
    print("Dest Stdev", dest_stds)
    print("Source Stdev", [(params[i] * source_stds[i]) - params[i] for i in range(3)])
    print("Dest Stdev", [(dest[i] * dest_stds[i] - dest[i]) for i in range(3)])

    predictions = []
    while len(predictions) < 1000:
        sample = []
        for i in range(3):
            sample.append(np.random.normal(params[i], 0.2 * params[i]))
        try:
            c_logs = [log(sample[i] / params[i]) for i in range(3)]
            predictions.append(exp(np.dot(coeff_means[0], c_logs)))
        except: pass

    print(np.mean(predictions))
    print(np.std(predictions))

    fig = plt.figure()
    plt.title("Latency predictions")
    plt.hist(predictions, 100)
    plt.xlabel("Latency")
    plt.ylabel("Count")
    maybe_plot()

    predictions = []
    while len(predictions) < 1000:
        sample = []
        for i in range(3):
            sample.append(np.random.normal(dest[i], 0.2 * dest[i]))
        try:
            c_logs = [log(sample[i] / dest[i]) for i in range(3)]
            predictions.append(exp(np.dot(coeff_means[0], c_logs)))
        except: pass

    print(np.mean(predictions))
    print(np.std(predictions))

    fig = plt.figure()
    plt.title("Latency predictions")
    plt.hist(predictions, 100)
    plt.xlabel("Latency")
    plt.ylabel("Count")
    maybe_plot()
'''

