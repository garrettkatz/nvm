from os import system, path
import numpy as np

num_tests = 500
num_faces = 5

#min_params = (0.0, 0.0, 0.0)
#max_params = (1.0, 5.0, 5.0)

min_params = (4.0, 0.0, 0.0)
max_params = (5.0, 5.0, 5.0)

tonic = 0.0                                                                     
Ra = 1.0                                                                        
Rv = 20.0 

for i in xrange(num_tests):
    filename = "./log/explore/log_%07d.txt" % i

    if not path.exists(filename):
        '''
        s = 0.0
        av = 0.0
        va = 0.0
        r = 0.1

        while r > 0.09:
            s = max(0.0, np.random.uniform(min_params[0], max_params[0]))
            av = max(0.0, np.random.uniform(min_params[1], max_params[1]))
            va = max(0.0, np.random.uniform(min_params[2], max_params[2]))
            r = Ra * (s - va * tonic) / (1 + (Ra * Rv * av * va)) 
        '''

        s = max(0.0, np.random.uniform(min_params[0], max_params[0]))
        av = max(0.0, np.random.uniform(min_params[1], max_params[1]))
        va = max(0.0, np.random.uniform(min_params[2], max_params[2]))
        r = Ra * (s - va * tonic) / (1 + (Ra * Rv * av * va)) 

        print("Running exploratory subject %02d (%f, %f, %f, steady=%f)" % (i, s, av, va, r))
        system("python visual_pathway.py -a %f -av %f -va %f -f %s -n %d > %s" %
            (s, av, va, filename.replace(".txt", ".png"), num_faces, filename))
