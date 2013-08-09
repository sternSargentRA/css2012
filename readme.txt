The attached zip file contains programs used to simulate the posterior for "Price Level 
Uncertainty and Stability in the UK" (Cogley, Sargent, and Surico 2012).

The main program is SWUC_model_SWRprior_2011.m.  This file loads data from 

* UKdata.xls
* Lindert_Williamson.txt
* Bowley.txt
* LaborDepartment.txt

After setting priors and defining arrays, it launches a Metropolis-within-Gibbs chain that 
calls a number of subroutines.

The files 

* kf_SWR.m 
* GIBBS1.m 

execute Carter and Kohn's (1994) forward filter and backword smoother, respectively.

The files 

* svmh0.m
* svmh.m
* svmhT.m 

execute the algorithm of Jacqier, et al for simulating stochastic volatilities.  The first subroutine
is for date 0, the last for date T, and the middle is for interior dates.

The file 

*ig2.m 

simulates a draw from an inverse-gamma density.
