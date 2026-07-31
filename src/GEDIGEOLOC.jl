"""
    GEDIGEOLOC

Bayesian inference for GEDI geolocation offsets within a contiguous beam
collection.

Each footprint's offset is drawn from a grid of candidates, scored against a
grid of reference (DEM) values, under a continuous-lag AR(1) prior along track.
See [`geolocate`](@ref).
"""
module GEDIGEOLOC

using LinearAlgebra
using Random
using Distributions
using Distances
using LogExpFunctions
using ProgressBars
using Roots

include("covariances.jl")

include("ar1.jl")
export AR1

include("grid.jl")
export OffsetGrid, nearestindex, centerindex, islattice

include("offsets.jl")

include("geoloc.jl")
export geolocate, offsets, posteriormean

include("predict.jl")
export predoffsets

include("misc.jl")
export gammashaperate, betaparams, logitmeansd

include("deprecated.jl")
export geolocate2, geolocate_gfe

end
