from math import log, exp
import numpy as np
from numpy import zeros, matrix
from scipy.linalg import inv, sqrtm


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
    else:
        h = hlast

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
        S0[:, i] = S1[:, i-1] + P1[:, :, i-1].dot(C.T) * (Y[i] - y10) / V10  # E(S(t|t))
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
    if isinstance(x, np.ndarray):
        T = x.size if x.ndim == 1 else x.shape[0]
    elif isinstance(x, float) or isinstance(x, int):
        T = 1
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
        SA[:, -i] = SM + np.real(sqrtm(P)).dot(wa[:, -i])

    return SA
