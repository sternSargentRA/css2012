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

if sys.version_info[0] >= 3:
    xrange = range

start_time = time()


##---------------------------- Function definitions
def svmhT(hlag, alpha, delta, sv, yt, hlast):
    """
    This function returns a draw from the posterior conditional density
    for the stochastic volatility parameter at time T. This is
    conditional on the lagging realization, hlag, as well as the data
    and parameters of the svol process.

    hlast is the previous draw in the chain, and is used in the acceptance step.
    R is a dummy variable that takes a value of 1 if the trial is rejected, 0 if accepted.

    Following JPR (1994), we use a MH step, but with a simpler log-normal proposal density.
    (Their proposal is coded in jpr.m.)
    h = svmhT(hlag, alpha, delta, sv, y, hlast)

    TODO: Clean up docstring

    VERIFIED (1x) SL (8-9-13)
    """
    # mean and variance for log(h) (proposal density)
    mu = alpha + delta * np.log(hlag)
    ss = sv ** 2.

    # candidate draw from lognormal
    htrial = np.exp(mu + (ss ** .5) * np.random.randn(1))

    # acceptance probability
    lp1 = -0.5 * log(htrial) - (yt ** 2) / (2 * htrial)
    lp0 = -0.5 * log(hlast) - (yt ** 2) / (2 * hlast)
    accept = min(1., exp(lp1 - lp0))

    u = np.random.rand(1)
    if u <= accept:
        h = htrial
        R = 0
    else:
        h = hlast
        R = 1

    return h


def svmh0(hlead, alpha, delta, sv, mu0, ss0):
    """
    This file returns a draw from the posterior conditional density
    for the stochastic volatility parameter at time 0.  This is conditional
    on the first period realization, hlead, as well as the prior and parameters
    of the svol process.

    mu0 and ss0 are the prior mean and variance.  The formulas simplify if these are
    given by the unconditional mean and variance implied by the state, but we haven't
    imposed this.  (allows for alpha = 0, delta = 1)

    Following JPR (1994), we use a MH step, but with a simpler log-normal proposal density.
    (Their proposal is coded in jpr.m.)

    Usage
    -----
    h = svmh0(hlead, alpha, delta, sv, mu0, ss0)

    VERIFIED (1x) SL (8-9-13)
    """
    # mean and variance for log(h) (proposal density)
    ssv = sv ** 2
    ss = ss0 * ssv / (ssv + (delta ** 2) * ss0)
    mu = ss * (mu0 / ss0 + delta * (np.log(hlead) - alpha) / ssv)

    # import pdb; pdb.set_trace()

    # draw from lognormal (accept = 1, since there is no observation)
    h = np.exp(mu + (ss ** .5) * np.random.randn(1))

    return h


def svmh(hlead, hlag, alpha, delta, sv, yt, hlast):
    """
    This file returns a draw from the posterior conditional density
    for the stochastic volatility parameter at time t.  This is conditional
    on adjacent realizations, hlead and hlag, as well as the data and parameters
    of the svol process.

    hlast is the previous draw in the chain, and is used in the acceptance step.
    R is a dummy variable that takes a value of 1 if the trial is rejected, 0 if accepted.

    Following JPR (1994), we use a MH step, but with a simpler log-normal proposal density.
    (Their proposal is coded in jpr.m.)

    h = svmh(hlead, hlag, alpha, delta, sv, y, hlast)

    TODO: Clean up docstring

    VERIFIED (1x) SL (8-9-13)
    """
    # mean and variance for log(h) (proposal density)
    mu = alpha*(1-delta) + delta*(np.log(hlead)+np.log(hlag)) / (1+delta**2)
    ss = (sv**2) / (1+delta**2)

    # candidate draw from lognormal
    htrial = np.exp(mu + (ss**.5) * np.random.randn(1))

    # acceptance probability
    lp1 = -0.5 * np.log(htrial) - (yt**2) / (2 * htrial)
    lp0 = -0.5 * np.log(hlast) - (yt**2) / (2 * hlast)
    accept = min(1, np.exp(lp1 - lp0))

    u = np.random.rand(1)
    if u <= accept:
        h = htrial
    else:
        h = hlast

    return h


def rmean(x):
    "this computes the recursive mean for a matrix x"

    N, NG = x.shape
    rm = zeros((NG, N))
    rm[0, :] = x[:, 0].T
    for i in range(1, NG):
        rm[i, :] = rm[i - 1, :] + (1 / i) * (x[:, i].T - rm[i - 1, :])

    return rm


def kf_SWR(Y, Q, R, Sm, SI, PI, T):
    """
    This file performs the forward kalman filter recursions for the
    Stock-Watson-Romer model.

    Y is inflation
    Q, R are the SW state innovation variances
    Sm is the standard deviation of the measurement error
    SI, PI are the initial values for the recursions, S(1|0) and P(1|0)
    T is the sample size

    Usage
    -----
    S0, P0, P1 = kf_SWR(Y, Q, R, Sm, SI, PI, T)

    Notes
    -----
    In this function I use np.matrix INTERNALLY. Everything coming out
    of this function is a numpy array.

    VERIFIED (1x) SL (8-9-13)
    """

    # current estimate of the state, S(t|t)
    S0 = zeros((2, T))

    # one-step ahead estimate of the state, S(t+1|t)
    S1 = zeros((2, T))

    # current estimate of the covariance matrix, P(t|t)
    P0 = zeros((2, 2, T))

    # one-step ahead covariance matrix, P(t+1|t)
    P1 = zeros((2, 2, T))

    # constant parameters
    A = np.array([[0, 1], [0, 1]])
    C = np.array([1, 0])

    # date 1
    #CHECKME: Check the rest of the function
    y10 = C.dot(SI)  # E(y(t|t-1)
    D = np.asarray(Sm[0])
    V10 = np.asarray(np.dot(C.dot(PI), C.T) + D.dot(D.T))  # V(y(t|t-1)
    S0[:, 0] = SI + PI.dot(C.T) * (Y[0] - y10) / V10  # E(S(t|t))
    P0[:, :, 0] = PI - (PI * matrix(C).T * C * PI) / V10  # V(S(t|t))
    S1[:, 0] = A.dot(S0[:, 0])  # E(S(t+1|t)
    B = np.array([[R[1] ** .5, Q[1] ** .5],
                  [0, Q[1] ** .5]])
    P1[:, :, 0] = np.dot(A.dot(P0[:, :, 0]), A.T) + B.dot(B.T)  # V(S(t+1|t)

    # Iterating through the rest of the sample
    for i in range(1, T):
        y10 = C.dot(S1[:, i-1])  # E(y(t|t-1)
        D = np.asarray(Sm[i])
        V10 = np.dot(C.dot(P1[:, :, i-1]), C.T) + D.dot(D.T)  # V(y(t|t-1)
        S0[:, i] = S1[:,i-1] + P1[:,:,i-1].dot(C.T) * (Y[i] - y10) / V10  # E(S(t|t))
        P0[:, :, i] = P1[:, :, i-1] - (P1[:, :, i-1] * matrix(C).T * C * P1[:, :, i-1]) / V10  # V(S(t|t))
        S1[:, i] = A.dot(S0[:, i])  # E(S(t+1|t))
        B = np.array([[R[i+1] ** .5, Q[i+1] ** .5],
                      [0, Q[i+1] ** .5]])
        P1[:, :, i] = np.dot(A.dot(P0[:, :, i]), A.T) + B.dot(B.T)  # V(S(t+1|t))

    return S0, P0, P1


def ig2(v0, d0, x):
    """
    This file returns posterior draw, v, from an inverse gamma with
    prior degrees of freedom v0/2 and scale parameter d0/2.  The
    posterior values are v1 and d1, respectively. x is a vector of
    innovations.

    The simulation method follows bauwens, et al p 317.  IG2(s,v)
        simulate x = chisquare(v)
        deliver s/x

    BUG: Should return scalar.
    """
    T = x.size if x.ndim == 1 else x.shape[0]
    v1 = v0 + T
    d1 = d0 + np.inner(x, x)
    z = np.random.randn(v1)
    x = np.inner(z, z)
    v = d1 / x
    return v


def gibbs1_swr(S0, P0, P1, T):
    """
    function SA = GIBBS1_SWR(S0,P0,P1,T);

    This file executes the Carter-Kohn backward sampler for the
    Stock-Watson-Romer model.

    S0, P0, P1 are outputs of the forward Kalman filter

    VERIFIED (1x) SL (8-9-13)
    """
    A = np.array([[0, 1], [0, 1]])

    # initialize arrays for Gibbs sampler
    SA = zeros((2, T))  # artificial states
    SM = zeros((2, 1))  # backward update for conditional mean of state vector
    PM = zeros((2, 2))  # backward update for projection matrix
    P = zeros((2, 2))  # backward update for conditional variance matrix
    wa = np.random.randn(2, T)  # draws for state innovations

    # Backward recursions and sampling
    # Terminal state
    SA[:, -1] = S0[:, -1] + np.real(sqrtm(P0[:, :, -1])).dot(wa[:, -1])

    # iterating back through the rest of the sample
    for i in range(2, T + 1):
        PM = np.dot(P0[:, :, -i].dot(A.T), inv(P1[:, :, -i]))
        P = P0[:, :, -i] - np.dot(PM.dot(A), P0[:, :, -i])
        SM = S0[:, -i] + PM.dot(SA[:, -i+1] - A.dot(S0[:, -i]))
        SA[:, -i] = SM + np.real(sqrtm(P)).dot(wa[:,-i])

    return SA

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
