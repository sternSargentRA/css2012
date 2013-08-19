import sys
from time import time
from math import sqrt, log, exp
import pandas as pd
import numpy as np
from numpy import matrix, ones, zeros
from numpy.linalg import inv
from scipy.linalg import sqrtm
from scipy.io import savemat
from numbapro import autojit
from css2012Funcs import (svmhT, svmh0, svmh, kf_SWR, ig2, gibbs1_swr)

if sys.version_info[0] >= 3:
    xrange = range

start_time = time()
##---------------------------- Main Course

NG = 5000  # number of draws from Gibbs sampler per data file
NF = 20

##----- Load data
# Load military data
A = pd.read_excel('../data/UKdata.xls', "Price Data",
                  index_col=0)['Close'].reset_index(drop=True).values
y = np.log(A[1:]) - np.log(A[:-1])
t = y.shape[0]
date = np.arange(t) + 1210

Y0 = y[511:581]
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
date = np.arange(t) + 1791
data = pd.DataFrame(y, index=date)
data.to_csv('all_data.csv')

##----- Set VAR properties
L = 0  # VAR lag order
YS = y[L: t]

##----- A weakly informative prior
# prior mean on initial value of state first element is \pi
SI = ones(2) * Y0.mean()

# prior variance on initial state
PI = np.array([[.15 ** 2, 0], [0, 0.025 ** 2]])

R0 = Y0.var(ddof=1)  # prior variance for SW transient innovations
Q0 = R0 / 25  # prior variance for trend innovations

df = 2  # prior degrees of freedom

##----- priors for sv (inverse gamma) (standard dev for volatility innovation)
# stock and watson's calibrated value adjusted for time aggregation
v0 = 10.
svr0 = 0.2236 * sqrt((v0 + 1) / v0)
svq0 = 0.2236 * sqrt((v0 + 1) / v0)
dr0 = v0 * (svr0 ** 2.)
dq0 = v0 * (svq0 ** 2.)

##----- prior variance for log R0, log Q0 (ballpark numbers)
ss0 = 5

##----- prior for measurement-error variance \sigma_m (prior is same for both
# periods)
vm0 = 7.
sm0 = 0.5 * sqrt(R0) * sqrt((vm0 + 1) / vm0)
dm0 = vm0 * (sm0 ** 2)

# after 1948, the measurement error has a standard deviation of 1 basis point.
# This is just to simplify programming
sm_post_48 = 0.0001

##----- initialize gibbs arrays
SA = zeros((NG, 2, t))  # draws of the state vector
QA = zeros((t+1, NG))  # stochastic volatilities for SW permanent innovation
RA = zeros((t+1, NG))  # stochastic volatilities for SW transient innovation
SV = zeros((NG, 2))  # standard error for log volatility innovations
SMV = zeros((NG, 3))  # standard error for measurement error

##----- initialize stochastic volatilities and measurement error variance
QA[:, 0] = Q0
RA[:, 0] = R0
SV[0, :] = [svr0, svq0]

# set up SMT
SMT = zeros(date.size)
SMT[:157] = sm0
SMT[157:] = sm_post_48
SMV[0, :] = sm0

##----- begin MCMC
for i_f in xrange(NF):
    for i_g in xrange(1, NG):

        S0, P0, P1 = kf_SWR(YS, QA[:,i_g-1], RA[:,i_g-1], SMT, SI, PI, t)
        SA[i_g, :, :] = gibbs1_swr(S0, P0, P1, t)

        # stochastic volatilities
        f = np.diff(np.column_stack([SI, SA[i_g, :, :]])).T

        # log R|sv,y and log Q|sv, y
        RA[0, i_g] = svmh0(RA[1, i_g - 1], 0, 1, SV[i_g-1, 0],
                           np.log(R0), ss0)
        QA[0, i_g] = svmh0(QA[1, i_g-1], 0, 1, SV[i_g-1, 1],
                           np.log(Q0), ss0)
        for i in range(1, t):
            RA[i, i_g] = svmh(RA[i+1, i_g-1], RA[i-1, i_g], 0, 1,
                              SV[i_g-1, 0], f[i-1, 0], RA[i, i_g-1])

            QA[i, i_g] = svmh(QA[i+1, i_g-1], QA[i-1, i_g], 0, 1,
                              SV[i_g-1, 1], f[i-1, 1], QA[i, i_g-1])

        # TODO: f.shape[0] == t. jl and ml use f[T,1] here.
        # TODO: Also check that QA/RA.shape[0] == t+1
        RA[-1, i_g] = svmhT(RA[-2, i_g], 0, 1, SV[i_g-1, 0], f[-1, 0],
                            RA[-1, i_g-1])

        QA[-1, i_g] = svmhT(QA[-2, i_g], 0, 1, SV[i_g-1, 1], f[-1, 1],
                            QA[-1, i_g-1])

        # svr
        lr = np.log(RA[:, i_g])
        er = lr[1:] - lr[:-1]  # random walk
        v = ig2(v0, dr0, er)
        SV[i_g, 0] = sqrt(v)

        #svq
        lq = np.log(QA[:, i_g])
        eq = lq[1:] - lq[:-1]  # random walk
        v = ig2(v0, dr0, eq)
        SV[i_g, 1] = sqrt(v)

        # measurement error
        em = YS - SA[i_g, 0, :]
        v1 = ig2(vm0, dm0, em[:60])  # measurement error 1791-1850 (Lindert-Williamson)
        v2 = ig2(vm0, dm0, em[60:124])  # measurement error 1851-1914 (Bowley)
        v3 = ig2(vm0, dm0, em[124:157])  # measurement error 1915-1947 (Labor Department)
        SMV[i_g, :] = np.array([v1, v2, v3]) ** .5
        SMT[:60] = SMV[i_g, 0]
        SMT[60:124] = SMV[i_g, 1]
        SMT[124:157] = SMV[i_g, 2]

        if i_g % 100 == 0:
            e_time = time() - start_time
            print("Iteration (%i, %i). Elapsed time: %.5f" % (i_f, i_g, e_time))

    if i_f < 10:
        num = '0' + str(i_f)
    else:
        num = str(i_f)
    f_name = './output/swuc_swrp_' + num + '.mat'

    SD = SA[0:-1:10, :, :]
    RD = RA[:, 0:-1:10]
    QD = QA[:, 0:-1:10]
    VD = SV[0:-1:10, :]
    MD = SMV[0:-1:10, :]

    data = {'SD': SD,
            'QD': QD,
            'RD': RD,
            'VD': VD,
            'MD': MD}

    savemat(f_name, data)

  # Re-initialize the Gibbs arrays as buffer for back step
    SA[0, :] = SA[-1, :]
    QA[:, 0] = QA[:, -1]
    RA[:, 0] = RA[:, -1]
    SV[0, :] = SV[-1, :]
    SMV[0, :] = SMV[-1, :]



# RA, S0, P0, P1 = data['RA'], data['SO'], data['P0'], data['P1']
