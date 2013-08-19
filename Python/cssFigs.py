import sys
import os
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from css2012Funcs import ig2

NF = 20
NB = 10
f_name = "./SimData/swuc_swrp_{0}.mat"

if len(sys.argv) > 1:
    args = sys.argv[1:]
else:
    args = [None]

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

##------ Load simulation data
# VA: standard deviation of innovations to log volatilities
# MA: standard deviation of pre-1948 measurement errors
mat = loadmat(f_name.format(str(NB)))

# Prepare VA
VD = mat["VD"]
P, N = VD.shape
VA = np.zeros((N, (NF-NB) * P))
VA[:, :P] = VD.T

# Prepare MA
MD = mat["MD"]
P, N = MD.shape
MA = np.zeros((N, (NF-NB) * P))
MA[:, :P] = MD.T

# Prepare RA, QA, SA1, SA2
RD = mat["RD"]
T, P = RD.shape

RA = np.zeros((T, (NF-NB) * P))
QA = np.zeros((T, (NF-NB) * P))
SA1 = np.zeros(((NF-NB) * P, T - 1))
SA2 = np.zeros(((NF-NB) * P, T - 1))

RA[:, :P] = RD
QA[:, :P] = mat["QD"]
SA1[:P, :] = mat["SD"][:, 0, :]
SA2[:P, :] = mat["SD"][:, 1, :]

# Fill in the rest of it!
for i in range(NB + 1, NF):
    k = i - NB + 1
    num = str(i) if i > 9 else '0' + str(i)
    mat_i = loadmat(f_name.format(num))
    VA[:, (k-1)*P:k*P] = mat_i["VD"].T
    MA[:, (k-1)*P:k*P] = mat_i["MD"].T
    RA[:, (k-1)*P:k*P] = mat_i["RD"]
    QA[:, (k-1)*P:k*P] = mat_i["QD"]
    SA1[(k-1)*P:k*P, :] = mat_i["SD"][:, 0, :]
    SA2[(k-1)*P:k*P, :] = mat_i["SD"][:, 1, :]


# prior for \sigma_r,\sigma_q
v0 = 10
svr0 = 0.2236 * sqrt((v0 + 1) / v0)  # S&W's value for time aggregation
dr0 = v0 * (svr0 ** 2)
svmc1 = np.array([sqrt(ig2(v0, dr0, 0)) for i in range(100000)])

##------ prior for measurement-error variance \sigma_m (prior is same for all periods)
vm0 = 7
R0 = .0651 ** 2
sm0 = 0.5 * sqrt(R0)*sqrt((vm0 + 1) / vm0)
dm0 = vm0 * (sm0 ** 2)
svmc2 = np.array([sqrt(ig2(vm0, dm0, 0)) for i in range(100000)])

##------ Histograms in figures 2 and 3
# Compute histograms
Np1, Xp1 = np.histogram(svmc1, 50, density=False)
Nt, Xt = np.histogram(VA[0, :], Xp1, density=False)
Ns, Xs = np.histogram(VA[1, :], Xp1, density=False)
Np2, Xp2 = np.histogram(svmc2, 50, density=False)
Nm1, Xm1 = np.histogram(MA[0, :], Xp2, density=False)
Nm2, Xm2 = np.histogram(MA[1, :], Xp2, density=False)
Nm3, Xm3 = np.histogram(MA[2, :], Xp2, density=False)

if 'plots' in args:
    fig2 = plt.figure()
    ax = plt.axes()
    ax.plot(Xp1[:-1], (Np1 * 1.0) / Np1.sum(), '--b', linewidth=2, label="Prior")
    ax.plot(Xt[:-1], (Nt * 1.0) / Nt.sum(), '-b', linewidth=2, label=r'Posterior $\sigma_r$')
    ax.plot(Xs[:-1], (Ns * 1.0) / Ns.sum(), '-r', linewidth=2, label=r'Posterior $\sigma_q$')
    ax.set_xlabel(r"$\sigma_r$, $\sigma_q$", fontsize=18)
    ax.legend(loc=1)

    fig3 = plt.figure()
    ax2 = plt.axes()
    ax2.plot(Xp2[:-1], (Np2 * 1.0) / Np2.sum(), '--b', linewidth=2)
    ax2.plot(Xm1[:-1], (Nm1 * 1.0) / Nm1.sum(), '-r', linewidth=2)
    ax2.plot(Xm2[:-1], (Nm2 * 1.0) / Nm2.sum(), '-g', linewidth=2)
    ax2.plot(Xm3[:-1], (Nm3 * 1.0) / Nm3.sum(), '-b', linewidth=2)
    ax2.set_xlabel(r"$\sigma_{im}$", fontsize=18)
    ax2.legend([r'Prior $\sigma_m$',
                r'Posterior $\sigma_m$ 1791-1850',
                r'Posterior $\sigma_m$ 1851-1914',
                r'Posterior $\sigma_m$ 1915-1947'], loc=0)

# NOTE: Alternative for generating bar plots of histogram in one shot using
#       plt.hist

# Np1, Xp1, _ = plt.hist(svmc1, 50, normed=True, stacked=True, histtype='bar')
# Nt, Xt, _ = plt.hist(VA[0, :], Xp1, normed=True, stacked=True, histtype='bar')
# Ns, Xs, _ = plt.hist(VA[1, :], Xp1, normed=True, stacked=True, histtype='bar')
# plt.legend(['Prior', r'Posterior $\sigma_r$', r'Posterior $\sigma_q$'],
#            loc=0)
# plt.xlabel(r"$\sigma_r$, $\sigma_q$", fontsize=18)
# plt.figure()
# Np2, Xp2, _ = plt.hist(svmc2, 50, normed=True, stacked=True, histtype='bar')
# Nm1, Xm1, _ = plt.hist(MA[0, :], Xp2, normed=True, stacked=True, histtype='bar')
# Nm2, Xm2, _ = plt.hist(MA[1, :], Xp2, normed=True, stacked=True, histtype='bar')
# Nm3, Xm3, _ = plt.hist(MA[2, :], Xp2, normed=True, stacked=True, histtype='bar')
# plt.xlabel(r'$\sigma_{im}$', fontsize=18)
# plt.legend([r'Prior $\sigma_m$',
#             r'Posterior $\sigma_m$ 1791-1850',
#             r'Posterior $\sigma_m$ 1851-1914',
#             r'Posterior $\sigma_m$ 1915-1947'], loc=0)

##----- Figure 4
# compute MA, QA, SA1, SA2
NMC = VA.shape[1]

# Sort and make SRA, SQA the same shape as SSA1 and SSA2
SRA = np.sort(RA[1:, :], axis=1)
SQA = np.sort(QA[1:, :], axis=1)
SSA1 = np.sort(SA1, axis=0).T
SSA2 = np.sort(SA2, axis=0).T
SAT = SSA1 - SSA2

# Get confidence intervals
ci_inds = np.array(NMC * np.array([0.25, 0.5, 0.75]), dtype=int)
CRA = SRA[:, ci_inds]
CQA = SQA[:, ci_inds]
CSA1 = SSA1[:, ci_inds]
CSA2 = SSA2[:, ci_inds]
CSAT = SAT[:, ci_inds]

if 'plots' in args:
    fig4, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(date, CSA2[:, 1], '-r', linewidth=2)
    ax[0, 0].plot(date, CSA2[:, 0], '--b', linewidth=2)
    ax[0, 0].plot(date, CSA2[:, 2], '--b', linewidth=2)
    ax[0, 0].legend(["Median", "Interquartile Range"], loc=2, fontsize=10)
    ax[0, 0].set_title(r"$\mu_{t}$", fontsize=16)
    ax[0, 0].set_xlim((1790, 2011))

    ax[0, 1].plot(date, CSAT[:, 1], '-r', linewidth=2)
    ax[0, 1].plot(date, CSAT[:, 0], '--b', linewidth=2)
    ax[0, 1].plot(date, CSAT[:, 2], '--b', linewidth=2)
    ax[0, 1].set_title(r"$\pi_{t} - \mu_t$", fontsize=16)
    ax[0, 1].set_xlim((1790, 2011))

    ax[1, 0].plot(date, np.sqrt(CQA[:, 1]), '-r', linewidth=2)
    ax[1, 0].plot(date, np.sqrt(CQA[:, 0]), '--b', linewidth=2)
    ax[1, 0].plot(date, np.sqrt(CQA[:, 2]), '--b', linewidth=2)
    ax[1, 0].set_title(r"$q_t^{1/2}$", fontsize=16)
    ax[1, 0].set_xlim((1790, 2011))

    ax[1, 1].plot(date, np.sqrt(CRA[:, 1]), '-r', linewidth=2)
    ax[1, 1].plot(date, np.sqrt(CRA[:, 0]), '--b', linewidth=2)
    ax[1, 1].plot(date, np.sqrt(CRA[:, 2]), '--b', linewidth=2)
    ax[1, 1].set_title(r"$r_t^{1/2}$", fontsize=16)
    ax[1, 1].set_xlim((1790, 2011))

    # Save all figures
    if 'save' in args:
        if not os.path.exists("Figures"):
            os.makedirs("Figures")
        fig2.savefig("./Figures/Figure2.png", format="png", dpi=400)
        fig2.savefig("./Figures/Figure2.eps", format="eps", dpi=400)
        fig3.savefig("./Figures/Figure3.png", format="png", dpi=400)
        fig3.savefig("./Figures/Figure3.eps", format="eps", dpi=400)
        fig4.savefig("./Figures/Figure4.png", format="png", dpi=400)
        fig4.savefig("./Figures/Figure4.eps", format="eps", dpi=400)
