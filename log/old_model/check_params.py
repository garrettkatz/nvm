import numpy as np

tonic = 0.0
Ra = 1.0
Rv = 20.0

std_dev_fraction = 5.0

min_latency = 66.6

params = [
    (0.4, 1.0, 1.0),
    (0.6, 0.25, 2.5),
]

def estimate_latency(x,y,z):
    return max(min_latency,
        242.8270673 * x - 121.2206744 * y - 48.16467577 * z + 207.6454472)

def ts_to_ms(ts):
    return 2.45 * ts + 120

for amygdala_sensitivity, amy_vmpfc, vmpfc_amy in params:
    print(amygdala_sensitivity, amy_vmpfc, vmpfc_amy)

    results = []
    bad = []
    latencies = []
    clipped = 0
    for _ in xrange(100000):
        a = max(0.0,
            np.random.normal(amygdala_sensitivity,
                amygdala_sensitivity / std_dev_fraction))
        av = max(0.0,
            np.random.normal(amy_vmpfc, amy_vmpfc / std_dev_fraction))
        va = max(0.0,
            np.random.normal(vmpfc_amy, vmpfc_amy / std_dev_fraction))

        r = Ra * (a - va * tonic) / (1 + (Ra * Rv * av * va))
        results.append(r)
        if r > 0.09:
            #print(a, av, va, r)
            bad.append(r)

        lat = estimate_latency(a, av, va)
        if lat == min_latency:
            clipped += 1
        latencies.append(lat)

    print("MAX", max(results))
    print("MIN", min(results))
    print("BAD", float(len(bad)) / len(results))
    print("Estimated latency:", np.mean(latencies), np.std(latencies))
    print("               ms:", ts_to_ms(np.mean(latencies)), ts_to_ms(np.std(latencies)) - 120)
    print("Clipped:", float(clipped) / len(latencies))
    print("")
