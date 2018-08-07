from os import system, path
import numpy as np

num_tests = 20
num_faces = 20

healthy_params = (0.4, 1.0, 1.0)
ptsd_params    = (0.6, 0.25, 2.5)


for i in xrange(num_tests):
    filename = "./log/healthy/log_%02d.txt" % i
    if not path.exists(filename):
        s = max(0.0, np.random.normal(healthy_params[0], healthy_params[0] / 5.0))
        av = max(0.0, np.random.normal(healthy_params[1], healthy_params[1] / 5.0))
        va = max(0.0, np.random.normal(healthy_params[2], healthy_params[2] / 5.0))
        print("Running healthy subject %02d (%f, %f, %f)" % (i, s, av, va))
        system("python visual_pathway.py -a %f -av %f -va %f -f %s -n %d  -noise> %s" %
            (s, av, va, filename.replace(".txt", ".png"), num_faces, filename))

    filename = "./log/ptsd/log_%02d.txt" % i
    if not path.exists(filename):
        s = max(0.0, np.random.normal(ptsd_params[0], ptsd_params[0] / 5.0))
        av = max(0.0, np.random.normal(ptsd_params[1], ptsd_params[1] / 5.0))
        va = max(0.0, np.random.normal(ptsd_params[2], ptsd_params[2] / 5.0))
        print("Running PTSD subject %02d (%f, %f, %f)" % (i, s, av, va))
        system("python visual_pathway.py -a %f -av %f -va %f -f %s -n %d  -noise> %s" %
            (s, av, va, filename.replace(".txt", ".png"), num_faces, filename))
