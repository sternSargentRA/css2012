require("css2012")

tic()
pmap(mcmc_loop, [1:NF])
tot_time = toc()

prinln("\n\nComputation completed. Total execution time: $tot_time")
