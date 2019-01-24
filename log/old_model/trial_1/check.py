import os

for d in ("healthy", "ptsd"):
    for f in os.listdir(d):
        if f.endswith(".txt"):
            count = 0

            for line in open("%s/%s" % (d, f)):
                if line.startswith("Setting face"):
                    count += 1

            if count != 34:
                print("%s/%s is invalid (%d count)" % (d, f, count))
