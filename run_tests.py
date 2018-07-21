from os import system, path

num_tests = 30

for i in xrange(num_tests):
  filename = "./log/healthy/log_%02d.txt" % i
  if not path.exists(filename):
      print("Running healthy subject %02d" % i)
      system("python visual_pathway.py -a 0.2 -f %s > %s" %
          (filename.replace(".txt", ".png"), filename))

for i in xrange(num_tests):
  filename = "./log/ptsd/log_%02d.txt" % i
  if not path.exists(filename):
      print("Running PTSD subject %02d" % i)
      system("python visual_pathway.py -a 0.5 -f %s > %s" %
          (filename.replace(".txt", ".png"), filename))
