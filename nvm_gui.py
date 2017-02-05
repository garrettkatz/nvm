import multiprocessing as mp
import os
from nvm import nvm

# def info(title):
#     print title
#     print 'module name:', __name__
#     if hasattr(os, 'getppid'):  # only available on Unix
#         print 'parent process:', os.getppid()
#     print 'process id:', os.getpid()

# def f(name, q):
#     info('function f')
#     print 'hello', name

# def nvm(string):
#     print(string)

# if __name__ == '__main__':
#     q = mp.Queue()
#     info('main line')
#     p = mp.Process(target=f, args=('bob',))
#     p.start()
#     p.join()

if __name__ == '__main__':
    
