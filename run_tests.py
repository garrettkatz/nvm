from os import system, path
import numpy as np

log_path = "./log/new_model/"

num_faces = 5

vmpfc_s = 0.5
av = 1.0

mins =           (0.1, 0.1, 0.1)
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

#### log_0.6_1.9_1.7.txt:   0.6000   1.9000   1.7000 139.8000 7467.0620 24373.9373   0.3682   0.6712 
#### log_0.7_1.2_1.3.txt:   0.7000   1.2000   1.3000 234.6000 9189.9287 21785.3571   0.5157   0.7454
# log_0.5_1.3_2.0.txt:   0.5000   1.3000   2.0000 145.4000 6708.6248 28094.9433   0.3505   0.7455 
# log_0.7_1.1_1.5.txt:   0.7000   1.1000   1.5000 215.8000 8444.2289 23902.4633   0.4744   0.7546
#
# python visual_pathway.py -amy_s 0.6 -vmpfc_s 0.500000 -av 1.000000 -va 1.9 -l 1.8 -n 20
# python visual_pathway.py -amy_s 0.63507 -vmpfc_s 0.500000 -av 1.000000 -va 1.370689 -l 1.42838257 -n 20
#
#healthy_params      = (0.6, 1.9, 1.7)
#ptsd_params         = (0.7, 1.2, 1.3)
#recovered_params    = (0.7, 1.2, 1.7)
#healthy_params      = (0.5, 1.3, 2.0)
#ptsd_params         = (0.7, 1.1, 1.5)
#recovered_params    = (0.7, 1.1, 2.0)
healthy_params      = (0.6, 1.9, 1.8)
ptsd_params         = (0.63507, 1.370689, 1.42838257)
recovered_params    = (0.63507, 1.370689, 1.8)

num_tests = 30
num_faces = 20

for i in xrange(num_tests):
    filename = log_path + "final/healthy/log_%02d.txt" % i
    if not path.exists(filename):
        amy_s = max(0.0, np.random.normal(healthy_params[0], healthy_params[0] / 5.0))
        va = max(0.0, np.random.normal(healthy_params[1], healthy_params[1] / 5.0))
        l = max(0.0, np.random.normal(healthy_params[2], healthy_params[2] / 5.0))

        print("Running healthy subject %02d (%f, %f, %f)" % (i, amy_s, va, l))
        command = ("python visual_pathway.py "
            "-amy_s %f -vmpfc_s %f -av %f -va %f -l %f -f %s -n %d > %s" %
            (amy_s, vmpfc_s, av, va, l, filename.replace(".txt", ".png"), num_faces, filename))
        print(command)
        system(command)

    filename = log_path + "final/ptsd/log_%02d.txt" % i
    if not path.exists(filename):
        amy_s = max(0.0, np.random.normal(ptsd_params[0], ptsd_params[0] / 5.0))
        va = max(0.0, np.random.normal(ptsd_params[1], ptsd_params[1] / 5.0))
        l = max(0.0, np.random.normal(ptsd_params[2], ptsd_params[2] / 5.0))

        print("Running PTSD subject %02d (%f, %f, %f)" % (i, amy_s, va, l))
        command = ("python visual_pathway.py "
            "-amy_s %f -vmpfc_s %f -av %f -va %f -l %f -f %s -n %d > %s" %
            (amy_s, vmpfc_s, av, va, l, filename.replace(".txt", ".png"), num_faces, filename))
        print(command)
        system(command)

    filename = log_path + "final/recovered/log_%02d.txt" % i
    if not path.exists(filename):
        amy_s = max(0.0, np.random.normal(recovered_params[0], recovered_params[0] / 5.0))
        va = max(0.0, np.random.normal(recovered_params[1], recovered_params[1] / 5.0))
        l = max(0.0, np.random.normal(recovered_params[2], recovered_params[2] / 5.0))

        print("Running recovered subject %02d (%f, %f, %f)" % (i, amy_s, va, l))
        command = ("python visual_pathway.py "
            "-amy_s %f -vmpfc_s %f -av %f -va %f -l %f -f %s -n %d > %s" %
            (amy_s, vmpfc_s, av, va, l, filename.replace(".txt", ".png"), num_faces, filename))
        print(command)
        system(command)
