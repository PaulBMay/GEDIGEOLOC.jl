using GEDIGEOLOC

using LinearAlgebra, SparseArrays
using Random, Distributions
using Plots

using NearestNeighbors

function makeGrid(halfGridWidth)
    k = (halfGridWidth*2 + 1)^2
    grid = zeros(k, 2)
    count = 1
    for i in (-halfGridWidth):halfGridWidth
        for j in (-halfGridWidth):halfGridWidth
            grid[count,:] = [i j]
            count += 1
        end
    end
    return grid
end

function valsFromGrid(offsets::Matrix, grid::Matrix, gridVals::Matrix)

    n = size(offsets, 1)

    tree = KDTree(grid')
    nnIndices = nn(tree, offsets')[1]

    vals = zeros(n)
    for i in 1:n
        vals[i] = gridVals[nnIndices[i], i]
    end

    return vals

end


###################

Random.seed!(96)

nfull = 2_000
np = 500
n = nfull - np

fpSpacing = 60
L = nfull*fpSpacing

# Make grid

halfGridWidth = 25
grid = makeGrid(halfGridWidth)
k = size(grid,1)

varGrid = 1
rangeGrid = 12

SigmaGrid = varGrid*GEDIGEOLOC.expcor(grid, rangeGrid)

SigmaGridChol = cholesky(SigmaGrid)

refValsfull = SigmaGridChol.L * randn(k, nfull)

scatter(grid[:,1], grid[:,2], zcolor = refValsfull[:,100])

# Make Offsets

locfull = 1.0*(fpSpacing*(1:nfull))[:,:]

rangeOffset = 5e+3
VarOffset = 50*[1 0.7; 0.7 1]
noiseOffset = 0.05
muOffset = [4.0, -6.0]

RhoOffset = noiseOffset*I(nfull) + (1 - noiseOffset)*GEDIGEOLOC.expcor(locfull, rangeOffset)

offsetsfull = muOffset' .+ cholesky(RhoOffset).L*randn(nfull,2)*cholesky(VarOffset).U

plot(1:nfull, offsetsfull[:,1])
plot(1:nfull, offsetsfull[:,2])

# Make Realizations

τtrue = [100.0, 1.0]
πztrue = 0.05
ztrue = BitVector(rand(Bernoulli(πztrue), nfull))

xfull = valsFromGrid(offsetsfull, grid, refValsfull)

yfull = xfull + sqrt.( 1 ./ τtrue[ztrue .+ 1] ) .* randn(nfull)

scatter(xfull,yfull)

gridcenterind = argmin(sum(grid.^2, dims = 2)[:,1])
curreffull = refValsfull[CartesianIndex.(fill(gridcenterind, nfull), 1:nfull)]

scatter(curreffull, yfull)

############################

testind = sort(sample(2:(nfull-1), np, replace = false))
trainind = setdiff(1:nfull, testind)

loc = locfull[trainind,:]
refVals = refValsfull[:,trainind]
y = yfull[trainind]
offsets = offsetsfull[trainind,:]

locp = locfull[testind,:]
refValsp = refValsfull[:,testind]
yp = yfull[testind]
offsetsp = offsetsfull[testind,:]

########################

priors = (α = [0.0, 100.0], τ₀ = gammashaperate(100, 10), τ₁ = gammashaperate(1, 0.5), πz = betaparams(0.05, 0.05), Σ = [50.0, diagm(fill(100.0, 2))], ϕ = gammashaperate(rangeOffset, 1_000), ρ = [3.0, 0.5])

initial = (w = zeros(n,2), Σ = diagm(fill(50.0, 2)), α = 0.0, ϕ = rangeOffset, ρ = 1 - noiseOffset, μ = muOffset, τ = τtrue, πz = πztrue, z = ztrue[trainind])

samples = geolocate_gfe(y, loc, grid, refVals, priors, initial, 1_000)

#=hmu = mean(samples.h, dims = 3)[:,:,1]

scatter(vec(hmu), vec(offsets))


Σmu = mean(samples.Σ, dims = 3)

scatter(samples.ϕ)
scatter(samples.ρ)

zmu = mean(samples.z, dims = 2)

scatter(zmu, ztrue)=#



###########################

os_ps, gidx_ps = predoffsets(samples, locp, loc, grid; every = 10)

os_mu = mean(os_ps, dims = 3)[:,:,1]

scatter(os_mu[:,1], offsetsp[:,1])
Plots.abline!(1, 0)

scatter(os_mu[:,2], offsetsp[:,2])
Plots.abline!(1, 0)
