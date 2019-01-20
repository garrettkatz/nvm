from os import system, path
import numpy as np

num_tests = 20
num_faces = 20

for i in xrange(num_tests):
    filename = "./log/prelim/log_%02d.txt" % i
    if not path.exists(filename):
        print("Running preliminary subject %02d" % i)
        system("python visual_pathway.py -a 0.0 -av 0.0 -va 0.0 -f %s -n %d -noise> %s" %
            (filename.replace(".txt", ".png"), num_faces, filename))
