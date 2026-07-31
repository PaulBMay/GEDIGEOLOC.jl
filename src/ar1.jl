# ---------------------------------------------------------------------------
# Tridiagonal LDLᵀ
# ---------------------------------------------------------------------------

"""
    TriLDL{T}(n)

Workspace for the ``LDL^\\top`` factorisation of a symmetric positive-definite
tridiagonal matrix: `L` unit lower bidiagonal, `D` diagonal.  Every operation is
`O(n)` and allocation free.

There is no fill-reducing permutation, so — unlike a sparse Cholesky — there is
no permuted coordinate system to get wrong.
"""
struct TriLDL{T<:AbstractFloat}
    d::Vector{T}   # diagonal of D
    l::Vector{T}   # subdiagonal of L, length n-1
end

TriLDL{T}(n::Integer) where {T<:AbstractFloat} =
    TriLDL(zeros(T, n), zeros(T, max(n - 1, 0)))

Base.size(F::TriLDL) = (length(F.d), length(F.d))

"""
    ldl!(F, Q)

Factor the SPD `SymTridiagonal` `Q` into `F`, in place.
"""
function ldl!(F::TriLDL, Q::SymTridiagonal)
    d, l = F.d, F.l
    n = length(d)
    @inbounds begin
        d[1] = Q.dv[1]
        for i in 2:n
            l[i-1] = Q.ev[i-1] / d[i-1]
            d[i] = Q.dv[i] - l[i-1] * Q.ev[i-1]
        end
    end
    return F
end

LinearAlgebra.logdet(F::TriLDL) = sum(log, F.d)

"""
    ldiv!(F::TriLDL, X)

Overwrite `X` with `Q \\ X`.  Accepts a vector or an `n × p` matrix.
"""
function LinearAlgebra.ldiv!(F::TriLDL, X::AbstractVecOrMat)
    d, l = F.d, F.l
    n = length(d)
    @inbounds for c in axes(X, 2)
        for i in 2:n                        # L y = b
            X[i, c] -= l[i-1] * X[i-1, c]
        end
        for i in 1:n                        # D z = y
            X[i, c] /= d[i]
        end
        for i in (n-1):-1:1                 # Lᵀ x = z
            X[i, c] -= l[i] * X[i+1, c]
        end
    end
    return X
end

"""
    whitensolve!(F::TriLDL, Z)

Overwrite `Z` with ``L^{-\\top} D^{-1/2} Z``.  Standard normal columns in give
`N(0, Q⁻¹)` columns out.
"""
function whitensolve!(F::TriLDL, Z::AbstractVecOrMat)
    d, l = F.d, F.l
    n = length(d)
    @inbounds for c in axes(Z, 2)
        for i in 1:n
            Z[i, c] /= sqrt(d[i])
        end
        for i in (n-1):-1:1
            Z[i, c] -= l[i] * Z[i+1, c]
        end
    end
    return Z
end

# ---------------------------------------------------------------------------
# Continuous-lag AR(1) plus nugget
# ---------------------------------------------------------------------------

"""
    AR1(t; phi, rho)

Along-track prior for the geolocation offsets of a contiguous beam collection at
the strictly increasing 1-D coordinates `t` (along-track distance, or time):

```
h_i = μ + w_i + ε_i,    cor(w_i, w_j) = exp(-|t_i - t_j| / ϕ)
Var(w_i) = ρ Σ,         Var(ε_i) = (1-ρ) Σ
```

so `Var(vec(h)) = Σ ⊗ C` with `C = ρK + (1-ρ)I`, `K` the AR(1) correlation
matrix.  `ρ ∈ (0,1]` is the fraction of offset variance in the smooth
component; `ρ = 1` is the nugget-free model.

This is exactly a Matérn ν = 1/2 field in 1-D, which is Markov, so `K⁻¹` is
tridiagonal and every quantity the sampler needs is `O(n)`:

* `Qpost = ρ⁻¹K⁻¹ + (1-ρ)⁻¹I`, factorised once per `(ϕ, ρ)`
* `C⁻¹ = (1-ρ)⁻¹I − (1-ρ)⁻²Qpost⁻¹`
* `logdet C = n log ρ + n log(1-ρ) + logdet K + logdet Qpost`
* `logdet K = Σᵢ log(1 - rᵢ²)`,   `rᵢ = exp(-(t_{i+1} - t_i)/ϕ)`

Lags need not be equally spaced, so gaps in a beam collection are handled
exactly rather than by pretending the footprints are evenly spaced.
"""
mutable struct AR1{T<:AbstractFloat}
    lag::Vector{T}                       # t[i+1] - t[i],  length n-1
    r::Vector{T}                         # exp(-lag/ϕ)
    s::Vector{T}                         # 1 / (1 - r²)
    Kinv::SymTridiagonal{T,Vector{T}}    # AR(1) precision, correlation scale
    Qpost::SymTridiagonal{T,Vector{T}}   # ρ⁻¹Kinv + (1-ρ)⁻¹I
    F::TriLDL{T}                         # factorisation of Qpost
    onesolve::Vector{T}                  # C⁻¹1  (depends only on ϕ, ρ)
    onesum::T                            # 1ᵀC⁻¹1
    logdetK::T
    logdetC::T
    ϕ::T
    ρ::T
    work::Matrix{T}                      # n × 2 scratch
end

function AR1(t::AbstractVector{<:Real}; phi::Real, rho::Real)
    T = float(promote_type(eltype(t), typeof(phi), typeof(rho)))
    n = length(t)
    n ≥ 2 || throw(ArgumentError("need at least two footprints"))
    lag = T.(diff(collect(t)))
    all(>(0), lag) || throw(ArgumentError(
        "`t` must be strictly increasing — sort the beam by along-track position first"))
    ar = AR1{T}(lag, similar(lag), similar(lag),
                SymTridiagonal(zeros(T, n), zeros(T, n - 1)),
                SymTridiagonal(zeros(T, n), zeros(T, n - 1)),
                TriLDL{T}(n), zeros(T, n), zero(T), zero(T), zero(T),
                T(phi), T(rho), zeros(T, n, 2))
    return update!(ar, phi, rho)
end

nfootprints(ar::AR1) = length(ar.Kinv.dv)
hasnugget(ar::AR1) = ar.ρ < 1

"""
    update!(ar::AR1, ϕ, ρ)

Recompute every cached quantity for a new range / nugget pair.  `O(n)`, no
allocation.
"""
function update!(ar::AR1{T}, ϕ::Real, ρ::Real) where {T}
    0 < ρ ≤ 1 || throw(DomainError(ρ, "ρ must lie in (0, 1]"))
    ϕ > 0 || throw(DomainError(ϕ, "ϕ must be positive"))
    ar.ϕ, ar.ρ = T(ϕ), T(ρ)

    lag, r, s = ar.lag, ar.r, ar.s
    dv, ev = ar.Kinv.dv, ar.Kinv.ev
    n = length(dv)

    logdetK = zero(T)
    @inbounds for i in eachindex(lag)
        ri = exp(-lag[i] / ar.ϕ)
        si = 1 / (1 - ri * ri)
        r[i], s[i] = ri, si
        ev[i] = -si * ri
        logdetK += log1p(-ri * ri)
    end
    @inbounds begin
        dv[1] = s[1]
        for i in 2:(n-1)
            dv[i] = s[i-1] + s[i] * r[i] * r[i]
        end
        dv[n] = s[n-1]
    end
    ar.logdetK = logdetK

    if hasnugget(ar)
        invρ, invnug = 1 / ar.ρ, 1 / (1 - ar.ρ)
        @. ar.Qpost.dv = dv * invρ + invnug
        @. ar.Qpost.ev = ev * invρ
        ldl!(ar.F, ar.Qpost)
        ar.logdetC = n * log(ar.ρ) + n * log1p(-ar.ρ) + logdetK + logdet(ar.F)
    else
        ar.logdetC = logdetK
    end

    # C⁻¹1 is used by the μ update every sweep and changes only with (ϕ, ρ).
    fill!(ar.onesolve, one(T))
    applyCinv!(ar.onesolve, ar)
    ar.onesum = sum(ar.onesolve)

    return ar
end

"""
    applyCinv!(X, ar::AR1)

Overwrite `X` (an `n`-vector or `n × p` matrix, `p ≤ 2`) with `C⁻¹X`.
"""
function applyCinv!(X::AbstractVecOrMat, ar::AR1)
    Y = _scratch(ar, X)
    copyto!(Y, X)
    if hasnugget(ar)
        a = 1 / (1 - ar.ρ)
        ldiv!(ar.F, Y)
        @. X = a * X - a * a * Y
    else
        mul!(X, ar.Kinv, Y)
    end
    return X
end

function _scratch(ar::AR1, X::AbstractVector)
    length(X) == size(ar.work, 1) ? view(ar.work, :, 1) : similar(X)
end
function _scratch(ar::AR1, X::AbstractMatrix)
    (size(X, 1) == size(ar.work, 1) && size(X, 2) ≤ size(ar.work, 2)) ?
        view(ar.work, :, axes(X, 2)) : similar(X)
end

"""
    crossprod!(ar::AR1, res, CinvRes)

Return the symmetric 2×2 matrix `resᵀ C⁻¹ res` as the tuple `(m11, m12, m22)`.
`CinvRes` is scratch of the same size as `res` and holds `C⁻¹res` on return.
"""
function crossprod!(ar::AR1, res::AbstractMatrix, CinvRes::AbstractMatrix)
    copyto!(CinvRes, res)
    applyCinv!(CinvRes, ar)
    r1, r2 = view(res, :, 1), view(res, :, 2)
    c1, c2 = view(CinvRes, :, 1), view(CinvRes, :, 2)
    return (dot(r1, c1), dot(r1, c2), dot(r2, c2))
end

"""
    loglik_offsets(ar::AR1, res, CinvRes, Σinv, logdetΣ)

Marginal log density of `res = h .- μ'` under `Var(vec(h)) = Σ ⊗ C`, with `w`
integrated out.  `Σinv` is `(s11, s12, s22)` of `Σ⁻¹`.

This is the target for the Metropolis update of `(ϕ, ρ)`.  That move is
*collapsed* over `w`, so `w` must be redrawn before it is used again — see the
sweep ordering in [`geolocate`](@ref).
"""
function loglik_offsets(ar::AR1, res::AbstractMatrix, CinvRes::AbstractMatrix,
                        Σinv::NTuple{3,Float64}, logdetΣ::Float64)
    n, p = size(res)
    m11, m12, m22 = crossprod!(ar, res, CinvRes)
    s11, s12, s22 = Σinv
    sse = s11 * m11 + 2 * s12 * m12 + s22 * m22     # tr(Σ⁻¹ resᵀC⁻¹res)
    return -0.5 * (n * logdetΣ + p * ar.logdetC + sse)
end

"""
    sample_w!(w, ar::AR1, res, ΣU, Z, rng)

Draw `w ~ p(w | h, μ, Σ, ϕ, ρ)` where `res = h .- μ'`.  `ΣU` is the upper
Cholesky factor of `Σ`; `Z` is `n × 2` scratch.

`vec(w) ~ N(vec(Qpost⁻¹ res / (1-ρ)), Qpost⁻¹ ⊗ Σ)` — `Σ` cancels out of the
mean.  With no nugget (`ρ = 1`) the map is deterministic: `w = h - μ`.
"""
function sample_w!(w::AbstractMatrix, ar::AR1, res::AbstractMatrix,
                   ΣU::AbstractMatrix, Z::AbstractMatrix, rng::AbstractRNG)
    copyto!(w, res)
    hasnugget(ar) || return w
    w ./= (1 - ar.ρ)
    ldiv!(ar.F, w)                       # conditional mean
    randn!(rng, Z)
    whitensolve!(ar.F, Z)                # columns ~ N(0, Qpost⁻¹)
    mul!(w, Z, ΣU, true, true)           # w .+= Z * ΣU
    return w
end
