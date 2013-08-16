#### css2012.jl: Julia implementation of Matlab file SWUC_model_SWRprior_2011.m
##---------------------------- Initial Setup
tic()
using DataFrames
using MAT

NG = 5000 # number of draws from Gibbs sampler per data file
NF = 20

##---------------------------- Function definitions
include("cssfuncs.jl")

##---------------------------- Load Data
# Load military data
A = readcsv("../data/UKdata.txt")[:, 2]
y =(log(A[2:end]) - log(A[1:end-1]))
T = size(y, 1)
date = 1210 + [0:1:T-1]

Y0 = y[512:581]  # 1721-1790 training sample
YS_1948_2011 = y[739:802] # 1948-2011

# TODO: This is a hack from Python... Makes life easier for now
y = readcsv("../Matlab/y.csv")

T = size(y, 1)
date = 1791 + [0:1:T-1]

##---------------------------- Main Course
## VAR lag order
L = 0
YS = y[1+L:T]'
X = ones(T-L,1)

## a weakly informative prior
SI = mean(Y0)*ones(2,1)  # prior mean on initial value of state; first element is \pi
PI = [.15^2 0; 0 0.025^2]  # prior variance on initial state

R0 = var(Y0)  # prior variance for SW transient innovations
Q0 = R0/25  # prior variance for trend innovations

df = 2  # prior degrees of freedom

# clear initial sample
T, N = size(YS')

## priors for sv (inverse gamma) (standard dev for volatility innovation)
v0 = 10
svr0 = 0.2236*sqrt((v0+1)/v0) # stock and watson's calibrated value adjusted for time aggregation
svq0 = 0.2236*sqrt((v0+1)/v0)
dr0 = v0*(svr0^2)
dq0 = v0*(svq0^2)

## prior variance for log R0, log Q0 (ballpark numbers)
ss0 = 5

# prior for measurement-error variance \sigma_m (prior is same for both
# periods)
vm0 = 7
sm0 = 0.5*sqrt(R0)*sqrt((vm0+1)/vm0)
dm0 = vm0*(sm0^2)
sm_post_48 = 0.0001 # after 1948, the measurement error has a standard deviation of 1 basis point. This is just to simplify programming

# initialize gibbs arrays
SA = zeros(NG,2,T)  # draws of the state vector
QA = zeros(T+1,NG)  # stochastic volatilities for SW permanent innovation
RA = zeros(T+1,NG)  # stochastic volatilities for SW transient innovation
SV = zeros(NG,2)  # standard error for log volatility innovations
SMV = zeros(NG,3)  # standard error for measurement error

# initialize stochastic volatilities and measurement error variance
QA[:, 1] = Q0*ones(T+1,1)
RA[:, 1] = R0*ones(T+1,1)
SV[1, :] = [svr0 svq0]
SMT = [sm0*ones(157,1); sm_post_48*ones(size(date[158:221,1]))]
SMV[1, :] = sm0

# msqeeze(A) = squeeze(A, find(([size(A)..]. ==1)))

for file = 1:NF
    for iter = 2:NG
        S0, P0, P1 = kf_SWR(YS, QA[:, iter-1], RA[:, iter-1], SMT, SI, PI, T)
        SA[iter,:,:] = reshape(gibbs1_swr(S0, P0, P1, T), (1, 2, T))

        # stochastic volatilities
        f = diff([SI squeeze(SA[iter,:,:], 1)]') # SW state innovations
        # log R|sv,y
        RA[1,iter] = svmh0(RA[2,iter-1],0,1,SV[iter-1,1],log(R0),ss0)[1][1]
        for t = 2:T
            RA[t,iter] = svmh(RA[t+1,iter-1],RA[t-1,iter],0,1,SV[iter-1,1],f[t-1,1],RA[t,iter-1])[1][1]
        end
        RA[T+1,iter] = svmhT(RA[T,iter],0,1,SV[iter-1,1],f[T,1],RA[T+1,iter-1])[1][1]

        # log Q|sv,y
        QA[1,iter] = svmh0(QA[2,iter-1],0,1,SV[iter-1,2],log(Q0),ss0)[1][1]
        for t = 2:T
            QA[t,iter] = svmh(QA[t+1,iter-1],QA[t-1,iter],0,1,SV[iter-1,2],f[t-1,2],QA[t,iter-1])[1][1]
        end
        QA[T+1,iter] = svmhT(QA[T,iter],0,1,SV[iter-1,2],f[T,2],QA[T+1,iter-1])[1][1]

        # svr
        lr = log(RA[:,iter])
        er = lr[2:T+1,1] - lr[1:T,1]  # random walk
        v = ig2(v0,dr0,er)[1][1]
        SV[iter,1] = v^.5

        # svq
        lq = log(QA[:,iter])
        eq = lq[2:T+1,1] - lq[1:T,1]  # random walk
        v = ig2(v0,dq0,eq)[1][1]
        SV[iter,2] = v^.5

        # measurement error
        em = YS - squeeze(SA[iter,1,:], 2)
        v1 = ig2(vm0,dm0,em[1,1:60]')[1][1] # measurement error 1791-1850 (Lindert-Williamson)
        v2 = ig2(vm0,dm0,em[1,61:124]')[1][1] # measurement error 1851-1914 (Bowley)
        v3 = ig2(vm0,dm0,em[1,125:157]')[1][1] # measurement error 1915-1947 (Labor Department)
        SMV[iter,:] = [v1 v2 v3].^.5
        SMT[1:60,1] = SMV[iter,1]*ones(60,1)
        SMT[61:124,1] = SMV[iter,2]*ones(64,1)
        SMT[125:157,1] = SMV[iter,3]*ones(33,1)

        if mod(iter, 100) == 0
            println("Iteration ($file, $iter)")
        end
    end

    # Prepare data to write out

    SD = SA[1:10:NG, :, :]
    RD = RA[:, 1:10:NG]
    QD = QA[:, 1:10:NG]
    VD = SV[1:10:NG, :]
    MD = SMV[1:10:NG, :]

    if file < 10
        f_name = "swuc_swrp_0$file.mat"
    else
        f_name = "swuc_swrp_$file.mat"
    end

    matwrite(f_name, {
             "SD" => SD,
             "QD" => QD,
             "RD" => RD,
             "VD" => VD,
             "MD" => MD
             })

    # reinitialize gibbs arrays (buffer for back step)
    SA[1,:] = SA[NG,:]
    QA[:,1] = QA[:,NG]
    RA[:,1] = RA[:,NG]
    SV[1,:] = SV[NG,:]
    SMV[1,:] = SMV[NG,:]

end
tot_time = toc()
