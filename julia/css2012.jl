#### css2012.jl: Julia implementation of Matlab file SWUC_model_SWRprior_2011.m
##---------------------------- Initial Setup
tic()
using DataFrames
using MAT

##---------------------------- Run Control parameters
# Folder for saving the data. Relative to this folder. Exclude trailing slash
output_dir = "SimData"

# Base file name to append numbers to. Leave $(num) and (:stuff) in there!
file_name = (:"swuc_swrp_$(num).mat")

skip = 100  # number of Gibbs draws to do before printing

NG = 5000 # number of draws from Gibbs sampler per data file
NF = 20  # Number of times to run the simulation

if !isdir(output_dir)
    mkdir("./$output_dir")
end

##---------------------------- Function definitions
include("cssfuncs.jl")  # Just include these functions from the other file.

##---------------------------- Load Data
# TODO: Is this the best way to have non-repetitive code inclusion? My only
#       hangup is that some variables that are used below aren't defined in
#       this file.
include("load_data.jl")

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

iter_time = time()
for file = 1:NF
    for iter = 2:NG
        S0, P0, P1 = kf_SWR(YS, QA[:, iter-1], RA[:, iter-1], SMT, SI, PI, T)
        SA[iter,:,:] = reshape(gibbs1_swr(S0, P0, P1, T), (1, 2, T))

        # stochastic volatilities
        f = diff([SI squeeze(SA[iter,:,:], 1)]') # SW state innovations

        updateRQ(iter, RA, SV, R0, ss0, f, T)   # update RA inplace
        updateRQ(iter, QA, SV, R0, ss0, f, T)   # update QA inplace

        SV[iter,1] = computeSV(iter, RA, v0, dr0)  # svr
        SV[iter,2] = computeSV(iter, QA, v0, dr0)  # svr

        # measurement error
        measurement_error(iter, YS, SA, vm0, dm0, SMV, SMT)

        if mod(iter, skip) == 0
            msg = "Iteration ($file, $iter). Time for last $skip iterations:"
            msg = "$msg $(time() - iter_time)"
            println(msg)
            iter_time = time()
        end
    end

    # Prepare data to write out

    SD = SA[1:10:NG, :, :]
    RD = RA[:, 1:10:NG]
    QD = QA[:, 1:10:NG]
    VD = SV[1:10:NG, :]
    MD = SMV[1:10:NG, :]

    num = file < 10 ? string("0", file) : file
    save_path = joinpath(pwd(), output_dir, eval(file_name))

    matwrite(save_path, {"SD" => SD, "QD" => QD, "RD" => RD, "VD" => VD,
                         "MD" => MD})

    # reinitialize gibbs arrays (buffer for back step)
    SA[1,:] = SA[NG,:]
    QA[:,1] = QA[:,NG]
    RA[:,1] = RA[:,NG]
    SV[1,:] = SV[NG,:]
    SMV[1,:] = SMV[NG,:]

end
tot_time = toc()
