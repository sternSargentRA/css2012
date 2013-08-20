import cython
import numpy as np
from numpy import zeros, matrix
from scipy.linalg import inv, sqrtm
from libc.math cimport exp, log, sqrt, fmin

cimport numpy as np
from rng import SimpleRNG

rng = SimpleRNG()

cpdef double[::1] randn1d(unsigned int n):
    return np.ascontiguousarray([rng.GetNormal() for i in range(n)])


cpdef double[:, ::1] randn2d(unsigned int m, unsigned int n):
    cdef unsigned int i, k
    cdef double[:, ::1] ret = np.empty((m, n))
    for k in range(n):
        for i in range(m):
            ret[i, k] = rng.GetNormal()
    return ret


cdef double inner1d(double[:] x, double[:] y):
    cdef unsigned int n = len(x)
    cdef unsigned int i
    cdef double res
    for i in range(n):
        res += x[i] * y[i]
    return res

# TODO: Write a pure Cython dot function. Might need to do dotmm and dotvm
#       for matrix-matrix and matrix-vector products, respectively


@cython.boundscheck(False)
@cython.wraparound(False)
cdef svmhT(double hlag,
          double alpha,
          double delta,
          double sv,
          double yt,
          double hlast):
    """
    This function returns a draw from the poster pior conditional density
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
    cdef double mu, ss
    mu = alpha + delta * log(hlag)
    ss = sv ** 2.

    # candidate draw from lognormal
    cdef double htrial
    htrial = exp(mu + (ss ** .5) * rng.GetNormal())

    # acceptance probability
    cdef double lp1, lp0, accept, u, h
    lp1 = -0.5 * log(htrial) - (yt ** 2) / (2 * htrial)
    lp0 = -0.5 * log(hlast) - (yt ** 2) / (2 * hlast)
    accept = fmin(1., exp(lp1 - lp0))

    u = rng.GetUniform()
    if u <= accept:
        h = htrial
    else:
        h = hlast

    return h

@cython.boundscheck(False)
@cython.wraparound(False)
cdef svmh0(double hlead,
           double alpha,
           double delta,
           double sv,
           double mu0,
           double ss0):
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
    cdef double ssv, ss, mu
    ssv = sv ** 2
    ss = ss0 * ssv / (ssv + (delta ** 2) * ss0)
    mu = ss * (mu0 / ss0 + delta * (log(hlead) - alpha) / ssv)

    # import pdb; pdb.set_trace()

    # draw from lognormal (accept = 1, since there is no observation)
    cdef double h
    h = exp(mu + (ss ** .5) * rng.GetNormal())

    return h

@cython.boundscheck(False)
@cython.wraparound(False)
cdef svmh(double hlead,
          double hlag,
          double alpha,
          double delta,
          double sv,
          double yt,
          double hlast):
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
    cdef double mu, ss
    mu = alpha*(1-delta) + delta*(log(hlead)+log(hlag)) / (1+delta**2)
    ss = (sv**2) / (1+delta**2)

    # candidate draw from lognormal
    cdef double htrial
    htrial = exp(mu + (ss**.5) * rng.GetNormal())

    # acceptance probability
    cdef double lp1, lp0, accept, h
    lp1 = -0.5 * log(htrial) - (yt**2) / (2 * htrial)
    lp0 = -0.5 * log(hlast) - (yt**2) / (2 * hlast)
    accept = fmin(1, exp(lp1 - lp0))

    u = rng.GetUniform()
    if u <= accept:
        h = htrial
    else:
        h = hlast

    return h


# cdef rmean(double[:, ::1] x):
#     "this computes the recursive mean for a matrix x"
#     cdef int N, NG, i
#     cdef double[:, ::1] rm

#     N, NG = x.shape
#     rm = zeros((NG, N))
#     rm[0, :] = x[:, 0].T
#     for i in range(1, NG):
#         rm[i, :] = rm[i - 1, :] + (1 / i) * (x[:, i].T - rm[i - 1, :])

#     return rm

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef kf_SWR(double[:] Y,
             double[:] Q,
             double[:] R,
             double[:] Sm,
             double[:] SI,
             double[:, ::1] PI,
             int T):
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
    cdef:
        double[:, ::1]  A, C2
        double[:] C
        np.ndarray[double, ndim=2, mode='c'] S0, S1
        np.ndarray[double, ndim=3, mode='c'] P0, P1

    # current estimate of the state, S(t|t)
    S0 = zeros((2, T))

    # one-step ahead estimate of the state, S(t+1|t)
    S1 = zeros((2, T))

    # current estimate of the covariance matrix, P(t|t)
    P0 = zeros((2, 2, T))

    # one-step ahead covariance matrix, P(t+1|t)
    P1 = zeros((2, 2, T))

    # constant parameters
    A = np.array([[0., 1.], [0., 1.]])
    C = np.array([1., 0.])
    C2 = np.array([[1., 0.]])

    cdef double y10 , D, V10
    cdef double[:, ::1] B
    # date 1
    #CHECKME: Check the rest of the function
    y10 = np.dot(SI, C.T)  # E(y(t|t-1)
    # D = np.asarray(Sm[0])
    D = Sm[0]
    # V10 = np.asarray(np.dot(C.dot(PI), C.T) + D.dot(D.T))  # V(y(t|t-1)
    V10 = np.dot(np.dot(C, PI), C.T) + D * D
    # V10 = np.dot(C, np.dot((PI), C.T)) + D**2
    S0[:, 0] = SI + (np.dot(PI, C.T) * float(Y[0] - y10) / V10).squeeze()  # E(S(t|t))
    P0[:, :, 0] = PI - np.dot(np.dot(PI, C2.T), np.dot(C2, PI)) / V10  # V(S(t|t))
    S1[:, 0] = np.dot(A, S0[:, 0])  # E(S(t+1|t)
    B = np.array([[R[1] ** .5, Q[1] ** .5],
                  [0, Q[1] ** .5]])
    P1[:, :, 0] = np.dot(np.dot(A, P0[:, :, 0]), A.T) + np.dot(B, B.T)  # V(S(t+1|t)

    # Iterating through the rest of the sample
    cdef int i
    for i in range(1, T):
        y10 = np.dot(C, S1[:, i-1])  # E(y(t|t-1)
        # D = np.asarray(Sm[i])
        D = Sm[i]
        # V10 = np.dot(C.dot(P1[:, :, i-1]), C.T) + D.dot(D.T)  # V(y(t|t-1)
        V10 = np.dot(np.dot(C, P1[:, :, i-1]), C.T) + D**2
        S0[:, i] = S1[:, i-1] + P1[:, :, i-1].dot(C.T) * (Y[i] - y10) / V10  # E(S(t|t))

        # V(S(t|t))
        P0[:, :, i] = P1[:, :, i-1] - (np.dot(np.dot(P1[:, :, i-1], C.T),
                                       np.dot(C, P1[:, :, i-1])) / V10)

        S1[:, i] = np.dot(A, S0[:, i])  # E(S(t+1|t))
        B = np.array([[R[i+1] ** .5, Q[i+1] ** .5],
                      [0, Q[i+1] ** .5]])
        P1[:, :, i] = np.dot(np.dot(A, P0[:, :, i]), A.T) + np.dot(B, B.T)  # V(S(t+1|t))

    return S0, P0, P1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double ig2(double v0, double d0, double[:] x):
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
    cdef double v, d1, xx
    cdef double[:] z
    cdef int T, v1

    T = x.size
    v1 = <int> v0 + T
    d1 = d0 + inner1d(x, x)
    z = randn1d(v1)
    xx = inner1d(z, z)
    v = d1 / xx
    return v


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ig2_scalar(int v0, double d0, double x):
    """
    This file returns posterior draw, v, from an inverse gamma with
    prior degrees of freedom v0/2 and scale parameter d0/2.  The
    posterior values are v1 and d1, respectively. x is a vector of
    innovations.

    The simulation method follows bauwens, et al p 317.  IG2(s,v)
        simulate x = chisquare(v)
        deliver s/x

    BUG: Should return scalar.

    Notes
    -----
    This function is needed in the plotting routines!
    """
    cdef double v, d1, xx
    cdef double[:] z
    cdef int v1
    cdef int T = 1

    v1 = <int> v0 + T
    d1 = d0 + x * x
    z = randn1d(v1)
    xx = np.inner(z, z)
    v = d1 / xx
    return v


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef gibbs1_swr(double[:, ::1] S0,
               double[:, :, ::1] P0,
               double[:, :, ::1] P1,
               int T):
    """
    function SA = GIBBS1_SWR(S0,P0,P1,T);

    This file executes the Carter-Kohn backward sampler for the
    Stock-Watson-Romer model.

    S0, P0, P1 are outputs of the forward Kalman filter

    VERIFIED (1x) SL (8-9-13)
    """
    cdef:
        double[:, ::1] A, PM, P, wa
        double[:] SM
        np.ndarray[double, ndim=2, mode='c'] SA
        unsigned int i

    A = np.array([[0., 1.], [0., 1.]])

    # initialize arrays for Gibbs sampler
    SA = zeros((2, T))  # artificial states
    SM = zeros(2)  # backward update for conditional mean of state vector
    PM = zeros((2, 2))  # backward update for projection matrix
    P = zeros((2, 2))  # backward update for conditional variance matrix
    wa = randn2d(2, T)  # draws for state innovations

    # Backward recursions and sampling
    # Terminal state
    SA[:, T-1] = S0[:, T-1] + np.dot(np.real(sqrtm(P0[:, :, T-1])), (wa[:, T-1]))

    # iterating back through the rest of the sample
    for i in range(1, T):
        PM = np.dot(np.dot(P0[:, :, T-i], A.T), inv(P1[:, :, T-i]))
        P = P0[:, :, T-i] - np.dot(np.dot(PM, A), P0[:, :, T-i])
        SM = S0[:, T-i-1] + np.dot(PM, SA[:, T-i] - np.dot(A, S0[:, T-i-1]))
        SA[:, T-i-1] = SM + np.dot(np.real(sqrtm(P)), wa[:, T-i-1])

    return SA




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef updateRQ(unsigned int i_g,
               np.ndarray[double, ndim=2, mode='c'] RQ,
               double[:, ::1] SV,
               double RQ0,
               double ss0,
               double[:, ::1] f,
               unsigned int t,
               unsigned int tm1):
    RQ[0, i_g] = svmh0(RQ[1, i_g - 1], 0, 1, SV[i_g-1, 0],
                       log(RQ0), ss0)

    cdef unsigned int i
    for i in range(1, t):
        RQ[i, i_g] = svmh(RQ[i+1, i_g-1], RQ[i-1, i_g], 0, 1,
                          SV[i_g-1, 0], f[i-1, 0], RQ[i, i_g-1])

    RQ[t, i_g] = svmhT(RQ[tm1, i_g], 0, 1, SV[i_g-1, 0], f[tm1, 0],
                       RQ[tm1, i_g-1])

    # No return because we just modified RQ in place


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double computeSV(unsigned int i_g,
                np.ndarray[double, ndim=2, mode='c'] RQ,
                double v0,
                double dr0,
                unsigned int t):
    cdef np.ndarray[double, ndim=1, mode='c'] lrq = np.log(RQ[:, i_g])
    cdef double[:] erq = lrq[1:] - lrq[:t]  # random walk
    cdef double v = ig2(v0, dr0, erq)
    return sqrt(v)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef measurement_error(unsigned int i_g,
                        np.ndarray[double, ndim=1, mode='c'] YS,
                        np.ndarray[double, ndim=3, mode='c'] SA,
                        double vm0,
                        double dm0,
                        np.ndarray[double, ndim=2, mode='c'] SMV,
                        np.ndarray[double, ndim=1, mode='c'] SMT):
    em = YS - SA[i_g, 0, :]
    v1 = ig2(vm0, dm0, em[:60])  # measurement error 1791-1850 (Lindert-Williamson)
    v2 = ig2(vm0, dm0, em[60:124])  # measurement error 1851-1914 (Bowley)
    v3 = ig2(vm0, dm0, em[124:157])  # measurement error 1915-1947 (Labor Department)
    SMV[i_g, :] = np.array([v1, v2, v3]) ** .5
    SMT[:60] = SMV[i_g, 0]
    SMT[60:124] = SMV[i_g, 1]
    SMT[124:157] = SMV[i_g, 2]

    # Again, no returns because we modify SMV and SMT in place
