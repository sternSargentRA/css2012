# Load military data
const data_dir = joinpath(dirname(dirname(@__FILE__)), "data")
A = readcsv(joinpath(data_dir, "UKdata.txt"))[:, 2]
y =(log(A[2:end]) - log(A[1:end-1]))
T = size(y, 1)
date = 1210 + [0:1:T-1]

Y0 = y[512:581]  # 1721-1790 training sample
YS_1948_2011 = y[739:802] # 1948-2011

# TODO: This is a hack from Python... Makes life easier for now
y = readcsv("../Matlab/y.csv")

T = size(y, 1)
date = 1791 + [0:1:T-1]
