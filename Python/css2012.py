from math import sqrt
import pandas as pd
import numpy as np

NG = 100000  # number of draws from Gibbs sampler per data file
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

# Load Lindert Williamson data
lnP = np.log(pd.read_csv('../data/Bowley.txt',
                         skiprows=3, sep='  ').dropna())
YS_1847_1914 = lnP.diff().values.ravel()[1:]

# Load Lindert Williamson data
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

##----- Set VAR properties
L = 0  # VAR lag order
YS = y[L: t - 1]

##----- A weakly informative prior
# prior mean on initial value of state; first element is \pi
SI = np.ones(2) * Y0.mean()

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
sm0 = 0.5 * sqrt(R0)  *sqrt((vm0 + 1) / vm0)
dm0 = vm0 * (sm0 * 2)

# after 1948, the measurement error has a standard deviation of 1 basis point.
# This is just to simplify programming
sm_post_48 = 0.0001

##----- initialize gibbs arrays
SA = np.zeros((NG, 2, t))  # draws of the state vector
QA = np.zeros((t+1, NG))  # stochastic volatilities for SW permanent innovation
RA = np.zeros((t+1, NG))  # stochastic volatilities for SW transient innovation
SV = np.zeros((NG, 2))  # standard error for log volatility innovations
SMV = np.zeros((NG, 3))  # standard error for measurement error

##----- initialize stochastic volatilities and measurement error variance
QA[:, 0] = Q0
RA[:, 0] = R0
SV[0, :] = [svr0, svq0]

# set up SMT
SMT = np.zeros(date.size)
SMT[:157] = sm0
SMT[157:] = sm_post_48
SMV[0, :] = sm0

##----- begin MCMC
