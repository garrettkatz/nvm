import cProfile
import pstats

def prof():
    import aas_scratch    

# prof()

cProfile.run('prof()','pstats')
p = pstats.Stats('pstats')
p.strip_dirs().sort_stats('cumulative').print_stats(20)
# p.strip_dirs().sort_stats('cumulative').print_callees(50)
