module GEDIGEOLOC

using LinearAlgebra
using Distances
using Random
using Distributions
using ProgressBars
using Plots
using SparseArrays
using LogExpFunctions

##############################

struct NNGP
    B::SparseMatrixCSC{Float64}
    F::Vector{Float64}
    Border::Vector{Int}
end

export NNGP

################################

include("covariances.jl")

#############################

include("nngp.jl")
export getnngp
export getnngp!
export formprec
export simnngp

#######################

include("geoloc.jl")
export geolocate

#########################

include("misc.jl")
export gammashaperate

end
