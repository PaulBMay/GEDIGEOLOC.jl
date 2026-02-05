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

n = 1000
fpSpacing = 60
L = n*fpSpacing

# Make grid

halfGridWidth = 25
grid = makeGrid(halfGridWidth)
k = size(grid,1)

varGrid = 1
rangeGrid = 12

SigmaGrid = varGrid*GEDIGEOLOC.expcor(grid, rangeGrid)

SigmaGridChol = cholesky(SigmaGrid)

refVals = SigmaGridChol.L * randn(k, n)

scatter(grid[:,1], grid[:,2], zcolor = refVals[:,100])

# Make Offsets

loc = 1.0*(fpSpacing*(1:n))[:,:]

rangeOffset = 5e+3
VarOffset = 50*[1 0.7; 0.7 1]
noiseOffset = 0.05
muOffset = [4.0, -6.0]

RhoOffset = noiseOffset*I(n) + (1 - noiseOffset)*GEDIGEOLOC.expcor(loc, rangeOffset)

offsets = muOffset' .+ cholesky(RhoOffset).L*randn(n,2)*cholesky(VarOffset).U

plot(1:n, offsets[:,1])
plot(1:n, offsets[:,2])

# Make Realizations

τtrue = [100.0, 1.0]
πztrue = 0.05
ztrue = BitVector(rand(Bernoulli(πztrue), n))

x = valsFromGrid(offsets, grid, refVals)

y = x + sqrt.( 1 ./ τtrue[ztrue .+ 1] ) .* randn(n)

scatter(x,y)

gridcenterind = argmin(sum(grid.^2, dims = 2)[:,1])
curref = refVals[CartesianIndex.(fill(gridcenterind, n), 1:n)]

scatter(curref, y)

############################

priors = (α = [0.0, 100.0], τ₀ = gammashaperate(100, 10), τ₁ = gammashaperate(1, 0.5), πz = betaparams(0.05, 0.05), Σ = [50.0, diagm(fill(100.0, 2))], ϕ = gammashaperate(rangeOffset, 1_000), ρ = [3.0, 0.5])

initial = (w = zeros(n,2), Σ = diagm(fill(100.0, 2)), α = 0.0, ϕ = rangeOffset, ρ = 1 - noiseOffset, μ = muOffset, τ = τtrue, πz = πztrue, z = ztrue)

samples = geolocate_gfe(y, loc, grid, refVals, priors, initial, 10)
samples = geolocate_gfe(y, loc, grid, refVals, priors, initial, 1_000)

hmu = mean(samples.h, dims = 3)[:,:,1]

scatter(vec(hmu), vec(offsets))


Σmu = mean(samples.Σ, dims = 3)

scatter(samples.ϕ)
scatter(samples.ρ)

zmu = mean(samples.z, dims = 2)

scatter(zmu, ztrue)
