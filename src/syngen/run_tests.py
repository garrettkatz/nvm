from os import system, path
import numpy as np

log_path = "./log/new_model/"

num_faces = 5

vmpfc_s = 0.5
av = 1.0

mins =           (0.1, 0.1, 1.0)
maxs =           (1.0, 2.0, 2.0)
bins = (tuple(int((10*maxs[i] - 10*mins[i])) + 1 for i in range(3)))

for amy_s in np.linspace(mins[0],maxs[0],bins[0]):
    for va in np.linspace(mins[1],maxs[1],bins[1]):
        for l in np.linspace(mins[2],maxs[2],bins[2]):
            filename = log_path + "explore/log_%3.1f_%3.1f_%3.1f.txt" % (amy_s, va, l)
            if not path.exists(filename):
                print("Running subject (%f, %f, %f)" % (amy_s, va, l))
                command = ("python visual_pathway.py "
                    "-amy_s %f -vmpfc_s %f -av %f -va %f -l %f -f %s -n %d > %s" %
                    (amy_s, vmpfc_s, av, va, l, filename.replace(".txt", ".png"), num_faces, filename))
                print(command)
                system(command)

#log_0.8_1.3_1.5.txt:   0.8000   1.3000   1.5000 166.6000 11110.6607 25931.9045   0.2489   0.7331 
#log_0.9_1.2_1.1.txt:   0.9000   1.2000   1.1000 278.6000 12426.6552 23631.4669   0.4140   0.7559 
healthy_params      = (0.8, 1.3, 1.5)
ptsd_params         = (0.9, 1.2, 1.1)
recovered_params    = (0.9, 1.2, 1.5)

num_tests = 30
num_faces = 20

for i in xrange(num_tests):
    filename = log_path + "final/healthy/log_%02d.txt" % i
    if not path.exists(filename):
        amy_s = max(0.0, np.random.normal(healthy_params[0], healthy_params[0] / 10.0))
        va = max(0.0, np.random.normal(healthy_params[1], healthy_params[1] / 10.0))
        l = max(0.0, np.random.normal(healthy_params[2], healthy_params[2] / 10.0))

        print("Running healthy subject %02d (%f, %f, %f)" % (i, amy_s, va, l))
        command = ("python visual_pathway.py "
            "-amy_s %f -vmpfc_s %f -av %f -va %f -l %f -f %s -n %d > %s" %
            (amy_s, vmpfc_s, av, va, l, filename.replace(".txt", ".png"), num_faces, filename))
        print(command)
        system(command)

    filename = log_path + "final/ptsd/log_%02d.txt" % i
    if not path.exists(filename):
        amy_s = max(0.0, np.random.normal(ptsd_params[0], ptsd_params[0] / 10.0))
        va = max(0.0, np.random.normal(ptsd_params[1], ptsd_params[1] / 10.0))
        l = max(0.0, np.random.normal(ptsd_params[2], ptsd_params[2] / 10.0))

        print("Running PTSD subject %02d (%f, %f, %f)" % (i, amy_s, va, l))
        command = ("python visual_pathway.py "
            "-amy_s %f -vmpfc_s %f -av %f -va %f -l %f -f %s -n %d > %s" %
            (amy_s, vmpfc_s, av, va, l, filename.replace(".txt", ".png"), num_faces, filename))
        print(command)
        system(command)

    filename = log_path + "final/recovered/log_%02d.txt" % i
    if not path.exists(filename):
        amy_s = max(0.0, np.random.normal(recovered_params[0], recovered_params[0] / 10.0))
        va = max(0.0, np.random.normal(recovered_params[1], recovered_params[1] / 10.0))
        l = max(0.0, np.random.normal(recovered_params[2], recovered_params[2] / 10.0))

        print("Running recovered subject %02d (%f, %f, %f)" % (i, amy_s, va, l))
        command = ("python visual_pathway.py "
            "-amy_s %f -vmpfc_s %f -av %f -va %f -l %f -f %s -n %d > %s" %
            (amy_s, vmpfc_s, av, va, l, filename.replace(".txt", ".png"), num_faces, filename))
        print(command)
        system(command)
