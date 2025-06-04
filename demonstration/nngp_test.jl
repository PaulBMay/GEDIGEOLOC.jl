using GEDIGEOLOC

using LinearAlgebra, SparseArrays
using Random, Distributions
using Plots
using BenchmarkTools

n = 20000
ϕ = 0.2

Random.seed!(96)

loc = collect(range(0, 1, n))[:,:]

m = 1

@time nngp = getnngp(loc, ϕ, m)
#Σ = GEDIGEOLOC.expcor(loc, ϕ)

@time Q = formprec(nngp)
#Qtrue = inv(Σ)

Q[1:5, 1:5]
#Qtrue[1:5, 1:5]

z = randn(n)


#w1 = cholesky(Σ).L * z
w2 = LowerTriangular(nngp.B) \ (sqrt.(nngp.F) .* z)


#plot(loc, w1)
plot!(loc, w2)

cor(w1, w2)

@btime cholesky(Q)

D = sparse_hcat(fill(1.0,n, 1), spdiagm(fill(1.0, n)))

DtD = D'*D


@time Qp = blockdiag(spdiagm([0.0]), Q) + DtD

@btime cholesky(Qp)