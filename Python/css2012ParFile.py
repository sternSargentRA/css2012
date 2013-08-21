from css2012 import *
from css2012Funcs import pprint, size, rank

##----- Compute how to split work across processes
# Compute which files to do on each process
f_per_proc = NF // size  # NF imported from css2012
rem = NF % f_per_proc

# Distribute extra work to leading processes
extras = np.zeros(size, dtype=int)
extras[:rem] = 1

# number of files per process
each = np.ones(size, dtype=int) * f_per_proc + extras

# compute starting and ending file numbers for each process.
# NOTE: ends is one more than what each process really does because python
#       cuts top index on range.
ends = (np.ones(size, dtype=int) * f_per_proc + extras).cumsum()
starts = ends - each

##----- begin MCMC
my_start = starts[rank]
my_end = ends[rank]

start_time = time()
for i_f in xrange(my_start, my_end):
    mcmc_loop(i_f, pprint)

tot_time = time() - start_time
print("\n\nComputation completed. Total execution time: %.3f" % tot_time)
