# TODO: I'd love to use Gadfly, but I can't figure out the histogram plot...
# using Gadfly
# using Winston
using DataFrames
using MAT

##---------------------------- Function definitions
include("cssfuncs.jl")  # Just include these functions from the other file.

##---------------------------- Load Data
# TODO: Is this the best way to have non-repetitive code inclusion? My only
#       hangup is that some variables that are used below aren't defined in
#       this file.
include("load_data.jl")

NB = 11
NF = 20


function load_item(file_name::ASCIIString, var::ASCIIString)
    file = matopen(file_name)
    return read(file, var)
end


##---------------------------- Prepare moments and data

# standard deviation of innovations to log volatilities
# load VA
VD = load_item("./SimData/swuc_swrp_$NB.mat", "VD")

P, N = size(VD)

VA = zeros(N,(NF-NB+1)*P)
VA[:, 1:P] = VD'
for i = NB+1:NF,
    j = i - NB + 1
    if i < 10
        f_name = "./SimData/swuc_swrp_0$i.mat"
    else
        f_name = "./SimData/swuc_swrp_$i.mat"
    end
    VD = load_item(f_name, "VD")
    VA[:,(j-1)*P+1:j*P] = VD'
end

NMC = size(VA,2)
# bound was used for sensitivity analysis at an early stage and has been
# deactivated
dd = ones(NMC,1) # indicator that equals 1 if the bound is satisfied


# standard deviation of pre-1948 measurement errors
# load MA
MD = load_item("./SimData/swuc_swrp_$NB.mat", "MD")
P, N = size(MD)
MA = zeros(N, (NF-NB+1)*P)
MA[:, 1:P] = MD'
for i = NB+1:NF
    j = i - NB + 1
    if i < 10
        f_name = "./SimData/swuc_swrp_0$i.mat"
    else
        f_name = "./SimData/swuc_swrp_$i.mat"
    end
    MD = load_item(f_name, "MD")
    MA[:,(j-1)*P+1:j*P] = MD'
end

# prior for \sigma_r,\sigma_q
v0 = 10
svr0 = 0.2236*sqrt((v0+1)/v0) # stock and watson's calibrated value adjusted for time aggregation
dr0 = v0*(svr0^2)
svmc1 = Float64[ig2(v0, dr0, 0)[1][1]^.5 for i=1:100000]

# prior for measurement-error variance \sigma_m (prior is same for all periods)
vm0 = 7;
R0 = .0651^2;
sm0 = 0.5*sqrt(R0)*sqrt((vm0+1)/vm0);
dm0 = vm0*(sm0^2);
svmc2 = Float64[ig2(vm0, dm0, 0)[1][1]^.5 for i=1:100000]

# compute histograms
Xp1, Np1 = hist(svmc1, 50)
Xt, Nt = hist(VA[1, :][:], Xp1)
Xs, Ns = hist(VA[2, :][:], Xp1)
Xp2, Np2 = hist(svmc2, 50)
Xm1, Nm1 = hist(MA[1, :][:], Xp2)
Xm2, Nm2 = hist(MA[2, :][:], Xp2)
Xm3, Nm3 = hist(MA[3, :][:], Xp2)

##---------------------------- Generate figures
# p1 = FramedPlot()
# setattr(p1, "xlabel", "\\sigma_r, \\sigma_q")
# l1 = Curve(Xp1[2:end], Np1/sum(Np1), "type", "dash", "color", "blue")
# setattr(l1, "label", "Prior")

# l2 = Curve(Xt[2:end], Nt/sum(Nt), "color", "blue"); setattr(l2, "label", "Posterior \\sigma_r")
# l3 = Curve(Xs[2:end], Ns/sum(Ns), "color", "red"); setattr(l3, "label", "Posterior \\sigma_q")
# leg = Legend( .5, .8, {l1, l2, l3} )
# add(p1, l1, l2, l3, leg)
# Winston.display(p1)
# add(p, Curve(Xp1, Np1/sum(Np1), "--b", "label", "Prior"))
# add(p, Curve(Xt, Nt/sum(Nt), "-b", "label", "Posterior \\sigma_r"))
# add(p, Curve(Xs, Ns/sum(Ns), "-r", "label", "Posterior \\sigma_q"))

# l1 = plot(Np1/sum(Np1), Xp1, "--b"); setattr(l1, "label", "Prior")
# l2 = plot(Nt/sum(Nt), Xt, "-b"); setattr(l1, "label", "Posterior \\sigma_r")
# l3 = plot(Ns/sum(Ns), Xs, "-r"); setattr(l1, "label", "Posterior \\sigma_q")
# add(p, l1); add(p, l2); add(p, l3)

