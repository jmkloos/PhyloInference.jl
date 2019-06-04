
using PhyloInference
using Random


Random.seed!(1)
include("feed_forward.jl")
Random.seed!(1)
println("Run Gibbs algorithm")
#VARS
# tstart = -100.0
# tnow = 0.0

# Xobs = [5,7,6,8]
# Yobs = [6,5,3,4]
# σ = 0.1
# h = 0.01
θ_x = 0.0
θ_y = 0.0
iter_amount = 100000

PhyloInference.main(tstart,tnow,xstart,ystart,σ,h,θ_x,θ_y,Xobs,Yobs,iter_amount,X,Y)
