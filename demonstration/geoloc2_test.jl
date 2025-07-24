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

time = collect(1:n) .* (60 / 7_000)

rangeOffset = 1.0
VarOffset = 50*[1 0.7; 0.7 1]
muOffset = [4, -6]

RhoOffset = GEDIGEOLOC.expcor(time[:,:], rangeOffset)

offsets = muOffset' .+ cholesky(RhoOffset).L*randn(n,2)*cholesky(VarOffset).U

plot(1:n, offsets[:,1])
plot(1:n, offsets[:,2])

# Make Realizations
epsilon = 0.0025

z = valsFromGrid(offsets, grid, refVals)

y = z + sqrt(epsilon)*randn(n)

scatter(z,y)

gridcenterind = argmin(sum(grid.^2, dims = 2)[:,1])
curref = refVals[CartesianIndex.(fill(gridcenterind, n), 1:n)]

scatter(curref, y)

############################


priors = (α = [0.0, 4.0], τ = gammashaperate(1/epsilon, 10), Σ = [50.0, VarOffset], ϕ = gammashaperate(rangeOffset, 0.1), ρ = logitmeansd(0.95, 0.97, 0.999))

initial = (w = copy(offsets), Σ = VarOffset, α = 0.0, ϕ = rangeOffset, ρ = 0.95, μ = muOffset, τ = 1/epsilon, h = copy(offsets))

samples = geolocate(y, time[:,:], grid, refVals, priors, initial, 5_000)
samples2 = geolocate2(y, time, grid, refVals, priors, initial, 5_000; fixϕ = false, fixall = false)

hmu = mean(samples.h, dims = 3)[:,:,1]
hmu2 = mean(samples2.h, dims = 3)[:,:,1]

scatter(vec(hmu), vec(offsets))

sum((hmu - offsets).^2)

scatter(vec(hmu2), vec(offsets))

sum((hmu2 - offsets).^2)


Σmu = mean(samples2.Σ, dims = 3)

scatter(samples.ϕ)
scatter(samples.ρ)
#######################

    function mylogsumexp(x)
        xmax = maximum(x)
        expxadj = exp.(x .- xmax)
        return log(sum(expxadj)) + xmax
    end 

function mine(x)

    return exp.(x .- mylogsumexp(x))

end

using LogExpFunctions

logprob = randn(2500)

result = softmax(logprob)

result2 = mine(logprob)

cor(result, result2)

using BenchmarkTools

@btime softmax(logprob)
@btime mine(logprob)


#######################################

function getlogitpars(a, b, p)

    la = logit(a)
    lb = logit(b)

    μ = (la + lb)/2

    f(σ) = cdf(Normal(μ, σ), lb) - cdf(Normal(μ, σ), la) - p

    σ_hat = find_zero(f, (1e-3, 10.0), Bisection())
    
    return [μ, σ_hat]

end


ρprior = getlogitpars(0.95, 0.98, 0.999)

samps = logistic.( rand(Normal(ρprior[1], ρprior[2]), 10_000) )

histogram(samps)