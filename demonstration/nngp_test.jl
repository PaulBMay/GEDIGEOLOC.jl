using GEDIGEOLOC

using LinearAlgebra, SparseArrays
using Random, Distributions
using Plots
using BenchmarkTools

###############


function formQar1(ρ::Vector{Float64})

    n = length(ρ) + 1
    ρsq = @. ρ^2
    scale = @. 1 / (1 - ρsq)

    d = similar(scale, n)
    d[1] = scale[1]
    @inbounds for i in 2:n-1
        d[i] = scale[i-1] + scale[i] * ρsq[i]
    end
    d[n] = scale[n-1]

    e = @. -scale * ρ

    return SymTridiagonal(d, e)
end

function formQar1!(Q::SymTridiagonal, ρ::Vector{Float64})

    n = length(ρ) + 1
    ρsq = @. ρ^2
    scale = @. 1 / (1 - ρsq)

    Q.dv[1] = scale[1]
    @inbounds for i in 2:n-1
        Q.dv[i] = scale[i-1] + scale[i] * ρsq[i]
    end
    Q.dv[n] = scale[n-1]

    Q.ev .= @. -scale * ρ

    return nothing

end


function ar1_precision_exponential(ρ)
    n = length(ρ) + 1  # total number of points                 # ρ[1] = exp(-h[1]/ϕ), length n-1
    α = 1 ./ (1 .- ρ.^2)         # α[i] = 1 / ((1 - ρ_i^2) σ^2)

    d = zeros(n)
    e = zeros(n - 1)

    # Main diagonal entries
    d[1] = 1 + α[1] * ρ[1]^2
    for i in 2:n-1
        d[i] = α[i-1] + α[i] * ρ[i]^2
    end
    d[n] = α[n-1]

    # Off-diagonal entries
    for i in 1:n-1
        e[i] = -α[i] * ρ[i]
    end

    return SymTridiagonal(d, e)
end


function apply_ar1_precision(ρ::Vector{Float64}, B::Matrix{Float64})
    n, k = size(B)
    @assert length(ρ) == n - 1

    ρsq = @. ρ^2
    α = @. 1 / (1 - ρsq)

    QdotB = zeros(n, k)

    # First row
    QdotB[1, :] .= (1 + α[1] * ρsq[1]) .* B[1, :] .- α[1] * ρ[1] .* B[2, :]

    # Interior rows
    for i in 2:n-1
        QdotB[i, :] .= α[i-1] .* B[i-1, :] .+
                       α[i] * ρsq[i] .* B[i, :] .-
                       α[i] * ρ[i] .* B[i+1, :]
    end

    # Last row
    QdotB[n, :] .= α[n-1] .* B[n, :] .- α[n-1] * ρ[n-1] .* B[n-1, :]

    return QdotB
end



###################

n = 100
ϕ = 0.1

Random.seed!(96)

time = sort(rand(n))
time = range(0, 1, n)

m = 1

nngp = getnngp(time, ϕ, m)

Q = formprec(nngp)


Q[1:4,1:4]
Q[(n-2):end,(n-2):end]

lags = time[2:n] - time[1:(n-1)]

ρ = exp.(-lags / ϕ)


@time Q2 = formQar1(ρ)

@time Q2 = ar1_precision_exponential(ρ)

sum(abs.(Q - Q2))

B = randn(n,2)


@time Q2 * B

@time Q * B


@time logdet(Q2)
@time logdet(Q)


@time formQar1!(Q2, ρ)

@btime formQar1($ρ)

@btime formQar1!($Q2, $ρ)