from __future__ import print_function
import os
import sys
from time import time
from math import sqrt
import pandas as pd
import numpy as np
from numpy import ones, zeros
from scipy.io import savemat
from css2012Funcs import (svmhT, svmh0, svmh, kf_SWR, ig2, gibbs1_swr,
                          updateRQ, computeSV, measurement_error)

if sys.version_info[0] >= 3:
    xrange = range

##----- Run Control parameters
NF = 20  # Number of times to run the MCMC process
NG = 5000  # number of draws from Gibbs sampler per data file
NGm = NG - 1  # constant used in calculations

# Folder for saving the data. Relative to this folder. Exclude trailing slash
output_dir = "SimData"

# Base file name to append numbers to. Leave {num} in there somewhere!
file_name = "swuc_swrp_{num}.mat"

skip = 100  # number of Gibbs draws to do before printing status

# Other params needed below, but not to be modified.
save_path = output_dir + os.path.sep + file_name

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

##----- Load data
# Load military data
A = pd.read_excel('../data/UKdata.xls', "Price Data",
                  index_col=0)['Close'].reset_index(drop=True).values
y = np.log(A[1:]) - np.log(A[:-1])
t = y.shape[0]
date = np.arange(t) + 1210

Y0 = y[511:581]

if not os.path.exists("all_data.csv"):  # Don't repeat of not necessary
    YS_1948_2011 = y[738:802]

    # Load Lindert Williamson data
    lnP = np.log(pd.read_csv('../data/Lindert_Williamson.txt',
                             skiprows=4, sep='  '))
    YS_1791_1850 = lnP.diff().values.ravel()[1:]

    # Load Bowley data
    lnP = np.log(pd.read_csv('../data/Bowley.txt',
                             skiprows=3, sep='  ').dropna())
    YS_1847_1914 = lnP.diff().values.ravel()[1:]

    # Load LaborDepartmetn data
    lnP = np.log(pd.read_csv('../data/LaborDepartment.txt',
                             skiprows=4, sep='  '))
    YS_1915_1947 = lnP.diff().values.ravel()[1:-1]

    # Stitch it all together
    y = np.concatenate([YS_1791_1850,
                       YS_1847_1914[4:],
                       YS_1915_1947,
                       YS_1948_2011])
    t = y.shape[0]
    tm1 = t - 1
    date = np.arange(t) + 1791
    data = pd.DataFrame(y, index=date, columns=['y'])
    data.to_csv('all_data.csv')

else:
    data = pd.read_csv("all_data.csv")
    y = data['y'].values
    date = data.index.values
    t = y.shape[0]
    tm1 = t - 1

##----- Set VAR properties and priors
L = 0  # VAR lag order
YS = y[L: t]

# prior mean on initial value of state first element is \pi
SI = ones(2) * Y0.mean()

# prior variance on initial state
PI = np.array([[.15 ** 2, 0], [0, 0.025 ** 2]])  # eq 4

R0 = Y0.var(ddof=1)  # prior variance for SW transient innovations
Q0 = R0 / 25  # prior variance for trend innovations

df = 2  # prior degrees of freedom

# priors for sv (inverse gamma) (standard dev for volatility innovation)
# v0 is stock and watson's calibrated value adjusted for time aggregation
v0 = 10.
svr0 = 0.2236 * sqrt((v0 + 1) / v0)
svq0 = 0.2236 * sqrt((v0 + 1) / v0)
dr0 = v0 * (svr0 ** 2.)
dq0 = v0 * (svq0 ** 2.)

# prior variance for log R0, log Q0 (ballpark numbers)
ss0 = 5.

# prior for measurement-error variance \sigma_m (prior is same for both
# periods)
vm0 = 7.
sm0 = 0.5 * sqrt(R0) * sqrt((vm0 + 1) / vm0)
dm0 = vm0 * (sm0 ** 2)

# after 1948, the measurement error has a standard deviation of 1 basis point.
# This is just to simplify programming
sm_post_48 = 0.0001

##----- Initialize gibbs, volatility, and measurement arrays
SA = zeros((NG, 2, t))  # draws of the state vector
QA = zeros((t+1, NG))  # stochastic volatilities for SW permanent innovation
RA = zeros((t+1, NG))  # stochastic volatilities for SW transient innovation
SV = zeros((NG, 2))  # standard error for log volatility innovations
SMV = zeros((NG, 3))  # standard error for measurement error

# initialize stochastic volatilities and measurement error variance
QA[:, 0] = Q0
RA[:, 0] = R0
SV[0, :] = [svr0, svq0]

# set up SMT
SMT = zeros(date.size)
SMT[:157] = sm0
SMT[157:] = sm_post_48
SMV[0, :] = sm0


##----- Define main mcmc function
def mcmc_loop(i_f, p_func=print):
    """
    Based on inputs in this file, run an entire simulation and save the
    data to a Matlab .mat file.

    Parameters
    ==========
    i_f : int
        An integer that is used to replace {num} in the `file_name`
        variable from above when saving the .mat file.

    p_func : function, optional(default=print)
        The function to be used to print updates. This defaults to the
        print function imported from __future__. This option is used
        so that the parallel code can alter the print message to specify
        which process is calling print.

    Returns
    =======
    None : all work is done internally and desired objects are saved.

    """
    start_time = time()
    iter_time = time()
    for i_g in xrange(1, NG):
        S0, P0, P1 = kf_SWR(YS, QA[:, i_g-1], RA[:, i_g-1], SMT, SI, PI, t)
        SA[i_g, :, :] = gibbs1_swr(S0, P0, P1, t)

        # stochastic volatilities
        f = np.diff(np.column_stack([SI, SA[i_g, :, :]])).T

        updateRQ(i_g, RA, SV, R0, ss0, f, t, tm1)   # update RA inplace
        updateRQ(i_g, QA, SV, Q0, ss0, f, t, tm1)   # update QA inplace

        SV[i_g, 0] = computeSV(i_g, RA, v0, dr0, t)  # svr
        SV[i_g, 1] = computeSV(i_g, QA, v0, dr0, t)  # svq

        # measurement error
        measurement_error(i_g, YS, SA, vm0, dm0, SMV, SMT)

        ##################################### Done breaking it up!

        if i_g % skip == 0:
            tot_time = time() - start_time
            i_time = time() - iter_time
            msg = "Iteration ({0}, {1}). Total time: {2:.5f}. "
            msg += "Time since last print: {3:.5f}"
            p_func(msg.format(i_f, i_g, tot_time, i_time))
            iter_time = time()

    # Add leading 0 for single digit file counts to be consistent with Matlab
    num = '0' + str(i_f) if i_f < 10 else str(i_f)
    f_name = save_path.format(num=num)

    SD = SA[0:NG-1:10, :, :]
    RD = RA[:, 0:NG-1:10]
    QD = QA[:, 0:NG-1:10]
    VD = SV[0:NG-1:10, :]
    MD = SMV[0:NG-1:10, :]

    data = {'SD': SD,
            'QD': QD,
            'RD': RD,
            'VD': VD,
            'MD': MD}

    savemat(f_name, data)

  # Re-initialize the Gibbs arrays as buffer for back step
    SA[0, :] = SA[NGm, :]
    QA[:, 0] = QA[:, NGm]
    RA[:, 0] = RA[:, NGm]
    SV[0, :] = SV[NGm, :]
    SMV[0, :] = SMV[NGm, :]

##----- Do MCMC calculations in series.
if __name__ == '__main__':
    # The if __name__  == '__main__' clause allows us to "run" this file
    # normally to do the simulation in series. However, it also allows
    # us to import the mcmc_loop function in the parallel driver,
    # reducing code repetition.
    #
    # To 'run' the file simply to `python css2012.py` from the command
    # prompt or do `run css2012` from the ipython prompt.

    start_time = time()
    for i_f in xrange(NF):
        mcmc_loop(i_f)

