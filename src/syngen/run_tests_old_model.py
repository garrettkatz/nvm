from os import system, path
import numpy as np

num_tests = 30
num_faces = 20

# First try
healthy_params = (0.3905, 4.3801, 0.4338)
ptsd_params = (0.5278, 2.2265, 0.6790)

# Second try
healthy_params = (0.7310, 1.8918, 2.4826)
ptsd_params = (0.8548, 0.6302, 2.9092)

# Third try
healthy_params = (0.4746, 1.3300, 0.5568)
ptsd_params = (0.7246, 0.6113, 0.8624)



for i in xrange(num_tests):
    filename = "./log/final/healthy/log_%02d.txt" % i
    if not path.exists(filename):
        s = max(0.0, np.random.normal(healthy_params[0], healthy_params[0] / 10.0))
        av = max(0.0, np.random.normal(healthy_params[1], healthy_params[1] / 10.0))
        va = max(0.0, np.random.normal(healthy_params[2], healthy_params[2] / 10.0))
        print("Running healthy subject %02d (%f, %f, %f)" % (i, s, av, va))
        system("python visual_pathway.py -a %f -av %f -va %f -f %s -n %d  -noise> %s" %
            (s, av, va, filename.replace(".txt", ".png"), num_faces, filename))

    filename = "./log/final/ptsd/log_%02d.txt" % i
    if not path.exists(filename):
        s = max(0.0, np.random.normal(ptsd_params[0], ptsd_params[0] / 10.0))
        av = max(0.0, np.random.normal(ptsd_params[1], ptsd_params[1] / 10.0))
        va = max(0.0, np.random.normal(ptsd_params[2], ptsd_params[2] / 10.0))
        print("Running PTSD subject %02d (%f, %f, %f)" % (i, s, av, va))
        system("python visual_pathway.py -a %f -av %f -va %f -f %s -n %d  -noise> %s" %
            (s, av, va, filename.replace(".txt", ".png"), num_faces, filename))
