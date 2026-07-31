# ---------------------------------------------------------------------------
# Scalar / small-block Gibbs components.
#
# These are plain top-level functions rather than closures over the sampler's
# locals: they compile once, stay type stable, and show up under their own names
# in a profile.
# ---------------------------------------------------------------------------

"""
    sample_alpha(y, curref, τobs, prior, rng)

`α | ·` with a `N(mean, 1/precision)` prior, `prior = (mean, precision)`.
Heteroscedastic in `τobs`, so it covers the mixture error model too.
"""
function sample_alpha(y, curref, τobs, prior, rng::AbstractRNG)
    prec = Float64(prior[2])
    num = Float64(prior[2]) * Float64(prior[1])
    @inbounds @simd for i in eachindex(y)
        prec += τobs[i]
        num += τobs[i] * (y[i] - curref[i])
    end
    return num / prec + randn(rng) / sqrt(prec)
end

"""
    sample_tau(y, α, curref, prior, rng)

Single-component observation precision, `prior = (shape, rate)`.
"""
function sample_tau(y, α, curref, prior, rng::AbstractRNG)
    sse = 0.0
    @inbounds @simd for i in eachindex(y)
        r = y[i] - α - curref[i]
        sse += r * r
    end
    return rand(rng, Gamma(prior[1] + length(y) / 2, 1 / (prior[2] + sse / 2)))
end

"""
    sample_tau_mixture!(τ, y, α, curref, z, prior0, prior1, rng)

Two-component observation precision for the ground-finding-error model:
`z[i] = false` is the well-behaved component, `z[i] = true` the inflated one.
"""
function sample_tau_mixture!(τ, y, α, curref, z, prior0, prior1, rng::AbstractRNG)
    n1 = 0
    sse0 = sse1 = 0.0
    @inbounds for i in eachindex(y)
        r = y[i] - α - curref[i]
        if z[i]
            n1 += 1
            sse1 += r * r
        else
            sse0 += r * r
        end
    end
    n0 = length(y) - n1
    τ[1] = rand(rng, Gamma(prior0[1] + n0 / 2, 1 / (prior0[2] + sse0 / 2)))
    τ[2] = rand(rng, Gamma(prior1[1] + n1 / 2, 1 / (prior1[2] + sse1 / 2)))
    return τ
end

"""
    sample_z!(z, y, α, curref, τ, πz, rng)

Mixture indicators.  Written as a single logistic of the log-odds, which is
stable without an explicit max-subtraction.
"""
function sample_z!(z, y, α, curref, τ, πz, rng::AbstractRNG)
    b = log(πz) - log1p(-πz) + 0.5 * (log(τ[2]) - log(τ[1]))
    dτ = 0.5 * (τ[1] - τ[2])
    @inbounds for i in eachindex(y)
        r = y[i] - α - curref[i]
        z[i] = rand(rng) < logistic(muladd(dτ, r * r, b))
    end
    return z
end

"""
    sample_pi(z, prior, rng)

Mixture weight with a `Beta(prior[1], prior[2])` prior.
"""
function sample_pi(z, prior, rng::AbstractRNG)
    s = count(z)
    return rand(rng, Beta(prior[1] + s, prior[2] + length(z) - s))
end

"""
    sample_Sigma(ar, res, CinvRes, prior, rng)

`Σ | ·` with an `InverseWishart(ν₀, ν₀Ψ)` prior, `prior = (ν₀, Ψ)`.  `w` is
integrated out via `C⁻¹`.
"""
function sample_Sigma(ar::AR1, res, CinvRes, prior, rng::AbstractRNG)
    m11, m12, m22 = crossprod!(ar, res, CinvRes)
    ν0 = Float64(prior[1])
    Ψ = ν0 .* prior[2]
    scale = Symmetric([Ψ[1, 1]+m11 Ψ[1, 2]+m12; Ψ[2, 1]+m12 Ψ[2, 2]+m22])
    return rand(rng, InverseWishart(ν0 + size(res, 1), Distributions.PDMats.PDMat(scale)))
end

"""
    sample_mu!(μ, h, ar, Σ, rng)

`μ | ·` under a flat prior: `N((1ᵀC⁻¹1)⁻¹1ᵀC⁻¹h, (1ᵀC⁻¹1)⁻¹Σ)`.  `C⁻¹1` is
cached on `ar` and only changes when `(ϕ, ρ)` do.
"""
function sample_mu!(μ, h, ar::AR1, Σ, rng::AbstractRNG)
    s = 1 / ar.onesum
    o = ar.onesolve
    m1 = s * dot(o, view(h, :, 1))
    m2 = s * dot(o, view(h, :, 2))
    l11 = sqrt(s * Σ[1, 1])
    l21 = s * Σ[2, 1] / l11
    l22 = sqrt(max(s * Σ[2, 2] - l21 * l21, 0.0))
    z1, z2 = randn(rng), randn(rng)
    μ[1] = m1 + l11 * z1
    μ[2] = m2 + l21 * z1 + l22 * z2
    return μ
end

# ---------------------------------------------------------------------------
# Adaptive random-walk Metropolis for the offset-prior parameters (ϕ, ρ)
# ---------------------------------------------------------------------------

"""
    RangeMH(dim; propsd, target)

Random-walk Metropolis state for `θ = (logit ρ, log ϕ)` (or whichever subset is
being sampled), with Robbins–Monro adaptation of the step size during burn-in.

The original code used a fixed proposal variance, which for these two strongly
correlated parameters is the difference between a chain that moves and one that
doesn't.
"""
mutable struct RangeMH
    logstep::Float64
    naccept::Int
    ntotal::Int
    nadapt::Int
    target::Float64
end

RangeMH(; propsd::Real = 0.05, target::Real = 0.3) =
    RangeMH(log(propsd), 0, 0, 0, Float64(target))

acceptrate(mh::RangeMH) = mh.ntotal == 0 ? 0.0 : mh.naccept / mh.ntotal

function adapt!(mh::RangeMH, accepted::Bool)
    mh.nadapt += 1
    γ = min(0.5, 5 / mh.nadapt)
    mh.logstep += γ * ((accepted ? 1.0 : 0.0) - mh.target)
    return mh
end

"""
    sample_range!(ar, arprop, mh, res, CinvRes, Σinv, logdetΣ, priors, sampleρ, sampleϕ, adapting, rng)

Metropolis update of `(ϕ, ρ)` on the `(log ϕ, logit ρ)` scale, targeting the
`w`-collapsed density.  Returns `(ar, arprop, accepted)`; the two `AR1` objects
are swapped on acceptance so no factorisation is ever copied.
"""
function sample_range!(ar::AR1, arprop::AR1, mh::RangeMH, res, CinvRes,
                       Σinv::NTuple{3,Float64}, logdetΣ::Float64, priors,
                       sampleρ::Bool, sampleϕ::Bool, adapting::Bool, rng::AbstractRNG)
    (sampleρ || sampleϕ) || return (ar, arprop, false)

    step = exp(mh.logstep)
    lϕ = log(ar.ϕ)
    lρ = sampleρ ? logit(ar.ρ) : 0.0
    lϕp = sampleϕ ? lϕ + step * randn(rng) : lϕ
    lρp = sampleρ ? lρ + step * randn(rng) : lρ
    ϕp = exp(lϕp)
    ρp = sampleρ ? logistic(lρp) : ar.ρ

    update!(arprop, ϕp, ρp)

    ll = loglik_offsets(ar, res, CinvRes, Σinv, logdetΣ)
    llp = loglik_offsets(arprop, res, CinvRes, Σinv, logdetΣ)

    ϕprior = Gamma(priors.ϕ[1], 1 / priors.ϕ[2])
    lp = logpdf(ϕprior, ar.ϕ)
    lpp = logpdf(ϕprior, ϕp)
    if sampleρ
        ρprior = Normal(priors.ρ[1], priors.ρ[2])   # prior is on the logit scale
        lp += logpdf(ρprior, lρ)
        lpp += logpdf(ρprior, lρp)
    end

    # Jacobian for ϕ only: the ρ prior is already specified on the logit scale.
    logratio = (llp + lpp + lϕp) - (ll + lp + lϕ)
    accepted = log(rand(rng)) < logratio

    mh.ntotal += 1
    accepted && (mh.naccept += 1)
    adapting && adapt!(mh, accepted)

    return accepted ? (arprop, ar, true) : (ar, arprop, false)
end

# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------

function _require(nt, keys::NTuple{N,Symbol}, what::String) where {N}
    missing_keys = filter(k -> !haskey(nt, k), keys)
    isempty(missing_keys) && return nothing
    throw(ArgumentError("$what is missing required field(s): " *
                        join(string.(missing_keys), ", ")))
end

_residual!(res, h, μ) = (res .= h .- transpose(μ); res)

function _fill_tauobs!(τobs, τ, z, mixture::Bool)
    if mixture
        t0, t1 = τ[1], τ[2]
        @inbounds for i in eachindex(τobs)
            τobs[i] = z[i] ? t1 : t0
        end
    else
        fill!(τobs, τ isa Number ? Float64(τ) : Float64(τ[1]))
    end
    return τobs
end

_sym2inv(Σ) = let d = Σ[1, 1] * Σ[2, 2] - Σ[1, 2] * Σ[2, 1]
    (Σ[2, 2] / d, -Σ[1, 2] / d, Σ[1, 1] / d)
end

# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------

"""
    geolocate(y, t, grid, reference, priors, initial, nsamps; kwargs...)

Infer GEDI geolocation offsets for one contiguous beam collection.

# Model

For footprint `i` at along-track coordinate `t[i]`, with candidate offsets
`grid` and reference (DEM) values `reference[j, i]` at offset `grid[j, :]`:

```
y_i    = α + reference[h_i, i] + e_i          e_i ~ N(0, 1/τ)
h_i    = μ + w_i + ε_i                        ε_i ~ N(0, (1-ρ)Σ)
cor(w_i, w_j) = exp(-|t_i - t_j| / ϕ)         Var(w_i) = ρΣ
```

`h_i` is constrained to the grid.  With `mixture = true`, `e_i` is instead a
two-component normal mixture with weight `πz` on the inflated component, which
absorbs ground-finding errors.  With `nugget = false`, `ρ ≡ 1` and `h` is itself
the AR(1) chain.

Because ν = 1/2 Matérn is Markov in 1-D, `K⁻¹` is tridiagonal and the entire
prior apparatus — factorisation, log-determinant, solves, and the `w` draw — is
`O(n)` per sweep.

# Arguments

* `y` — `n` observed elevations.
* `t` — `n` strictly increasing along-track coordinates (distance or time).
* `grid` — `k × 2` candidate offsets, or an [`OffsetGrid`](@ref).
* `reference` — `k × n` reference values.  May be `Float32`; that halves the
  memory traffic of the dominant loop.
* `priors` — `NamedTuple` with `α = (mean, precision)`, `Σ = (ν₀, Ψ)`,
  `ϕ = (shape, rate)`; `τ = (shape, rate)` or, for `mixture = true`,
  `τ₀`, `τ₁` and `πz = (a, b)`; and `ρ = (mean, sd)` on the **logit** scale.
* `initial` — `NamedTuple` with at least `Σ, α, ϕ, μ, τ`; optionally
  `ρ, w, h, z, πz`.
* `nsamps` — number of draws to *store*.

# Keywords

* `mixture = false` — two-component observation error.
* `nugget = true` — include `ε_i`; `false` fixes `ρ = 1`.
* `sampleϕ = true`, `sampleρ = true` — update the prior parameters.
* `burn = 0`, `thin = 1` — `burn + nsamps*thin` sweeps are run.
* `window = Inf` — see the note below.
* `nchunks = Threads.nthreads()` — parallel chunks for the offset update.
* `propsd = 0.05`, `adapt = true` — Metropolis step size and its adaptation.
* `store_h = true`, `store_w = true` — the offset indices are always returned;
  these control the much larger coordinate arrays.
* `rng`, `progress`.

# The `window` keyword

Setting `window = w` restricts each footprint's categorical draw to the lattice
cells within `w` prior standard deviations of that footprint's prior mean, which
cuts the dominant loop by the ratio of the window area to the grid area.

This is an **approximation**: it renormalises the conditional over the window
instead of the full grid.  The dropped cells have negligible *prior* mass, but
the reference likelihood can be sharp enough to favour a far cell, so a jump the
exact sampler would make can be suppressed.  `w = 6` is a reasonable starting
point; compare against `window = Inf` on a subset before relying on it.

# Sweep order

`h → α → τ → (z, πz) → Σ → μ → (ϕ, ρ) → w`.

The `Σ`, `μ` and `(ϕ, ρ)` updates all target densities with `w` integrated out,
so `w` **must** be redrawn after them and before it is used again.  (Drawing `w`
before the `(ϕ, ρ)` update, as an earlier version did, leaves the chain with the
wrong stationary distribution.)

# Returns

A `NamedTuple` of draws: `hidx` (`n × nsamps` grid indices), `h`, `w`, `α`, `τ`,
`z`, `πz`, `Σ`, `μ`, `ρ`, `ϕ`, and `accept` (the Metropolis acceptance rate).
"""
function geolocate(y::AbstractVector, t::AbstractVecOrMat, grid,
                   reference::AbstractMatrix, priors, initial, nsamps::Integer;
                   mixture::Bool = false,
                   nugget::Bool = true,
                   sampleϕ::Bool = true,
                   sampleρ::Bool = true,
                   burn::Integer = 0,
                   thin::Integer = 1,
                   window::Real = Inf,
                   nchunks::Integer = Threads.nthreads(),
                   propsd::Real = 0.05,
                   adapt::Bool = true,
                   store_h::Bool = true,
                   store_w::Bool = true,
                   progress::Bool = true,
                   rng::AbstractRNG = Random.default_rng())

    g = OffsetGrid(grid)
    n, k = length(y), length(g)
    size(reference) == (k, n) || throw(DimensionMismatch(
        "reference must be $(k) × $(n) (grid points × footprints), got $(size(reference))"))
    length(t) == n || throw(DimensionMismatch("t must have length $(n), got $(length(t))"))
    nsamps ≥ 1 || throw(ArgumentError("nsamps must be positive"))
    thin ≥ 1 || throw(ArgumentError("thin must be positive"))

    _require(priors, (:α, :Σ, :ϕ), "priors")
    _require(initial, (:Σ, :α, :ϕ, :μ, :τ), "initial")
    if mixture
        _require(priors, (:τ₀, :τ₁, :πz), "priors (mixture = true)")
    else
        _require(priors, (:τ,), "priors")
    end
    sampleρ = sampleρ && nugget
    nugget && sampleρ && _require(priors, (:ρ,), "priors (sampleρ = true)")

    # --- state ---
    ρ0 = nugget ? Float64(get(initial, :ρ, 0.95)) : 1.0
    nugget && (0 < ρ0 < 1 || throw(ArgumentError("initial.ρ must lie in (0,1) when nugget = true")))
    ar = AR1(Float64.(vec(t)); phi = Float64(initial.ϕ), rho = ρ0)
    arprop = AR1(Float64.(vec(t)); phi = Float64(initial.ϕ), rho = ρ0)

    α = Float64(initial.α)
    μ = Float64[initial.μ[1], initial.μ[2]]
    Σ = Matrix{Float64}(initial.Σ)
    τ = mixture ? Float64[initial.τ[1], initial.τ[2]] : Float64[initial.τ isa Number ? initial.τ : initial.τ[1]]
    πz = Float64(get(initial, :πz, 0.05))
    z = BitVector(undef, n)
    if mixture && haskey(initial, :z)
        z .= initial.z
    else
        fill!(z, false)
    end

    w = zeros(Float64, n, 2)
    haskey(initial, :w) && copyto!(w, initial.w)

    h = zeros(Float64, n, 2)
    hidx = Vector{Int32}(undef, n)
    if haskey(initial, :h)
        @inbounds for i in 1:n
            hidx[i] = nearestindex(g, initial.h[i, 1], initial.h[i, 2])
        end
    else
        fill!(hidx, Int32(centerindex(g)))
    end
    curref = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        h[i, 1] = g.coords[hidx[i], 1]
        h[i, 2] = g.coords[hidx[i], 2]
        curref[i] = reference[hidx[i], i]
    end

    ws = OffsetWorkspace(k, n; nchunks = nchunks, rng = rng)
    _fill_tauobs!(ws.τobs, τ, z, mixture)
    mh = RangeMH(; propsd = propsd)

    res = zeros(Float64, n, 2)
    CinvRes = zeros(Float64, n, 2)
    Z = zeros(Float64, n, 2)

    # --- storage ---
    M = Int(nsamps)
    hidxs = Matrix{Int32}(undef, n, M)
    hs = store_h ? Array{Float64}(undef, n, 2, M) : nothing
    wsamp = (store_w && nugget) ? Array{Float64}(undef, n, 2, M) : nothing
    αs = Vector{Float64}(undef, M)
    τs = mixture ? Matrix{Float64}(undef, 2, M) : Matrix{Float64}(undef, 1, M)
    zs = mixture ? BitMatrix(undef, n, M) : nothing
    πzs = mixture ? Vector{Float64}(undef, M) : nothing
    Σs = Array{Float64}(undef, 2, 2, M)
    μs = Matrix{Float64}(undef, 2, M)
    ρs = Vector{Float64}(undef, M)
    ϕs = Vector{Float64}(undef, M)

    win = Float64(window)
    total = Int(burn) + M * Int(thin)
    iter = progress ? ProgressBar(1:total) : 1:total

    # --- sweep ---
    for m in iter
        sample_offsets!(hidx, h, curref, ws, g, reference, y, α, μ, w, Σ, ar; window = win)

        α = sample_alpha(y, curref, ws.τobs, priors.α, rng)

        if mixture
            sample_tau_mixture!(τ, y, α, curref, z, priors.τ₀, priors.τ₁, rng)
            sample_z!(z, y, α, curref, τ, πz, rng)
            πz = sample_pi(z, priors.πz, rng)
        else
            τ[1] = sample_tau(y, α, curref, priors.τ, rng)
        end
        _fill_tauobs!(ws.τobs, τ, z, mixture)

        _residual!(res, h, μ)
        Σ = sample_Sigma(ar, res, CinvRes, priors.Σ, rng)

        sample_mu!(μ, h, ar, Σ, rng)
        _residual!(res, h, μ)

        Σinv = _sym2inv(Σ)
        logdetΣ = log(Σ[1, 1] * Σ[2, 2] - Σ[1, 2] * Σ[2, 1])
        ar, arprop, _ = sample_range!(ar, arprop, mh, res, CinvRes, Σinv, logdetΣ,
                                      priors, sampleρ, sampleϕ, adapt && m ≤ burn, rng)

        # w is redrawn last: every update above is collapsed over it.
        ΣU = cholesky(Symmetric(Σ)).U
        sample_w!(w, ar, res, ΣU, Z, rng)

        if m > burn && (m - burn) % thin == 0
            s = (m - burn) ÷ thin
            @inbounds hidxs[:, s] .= hidx
            store_h && (@inbounds hs[:, :, s] .= h)
            wsamp !== nothing && (@inbounds wsamp[:, :, s] .= w)
            αs[s] = α
            τs[1, s] = τ[1]
            if mixture
                τs[2, s] = τ[2]
                @inbounds zs[:, s] .= z
                πzs[s] = πz
            end
            @inbounds Σs[:, :, s] .= Σ
            @inbounds μs[:, s] .= μ
            ρs[s] = ar.ρ
            ϕs[s] = ar.ϕ
        end
    end

    return (hidx = hidxs,
            h = hs,
            w = wsamp,
            α = αs,
            τ = mixture ? τs : vec(τs),
            z = zs,
            πz = πzs,
            Σ = Σs,
            μ = μs,
            ρ = ρs,
            ϕ = ϕs,
            grid = g,
            accept = acceptrate(mh))
end

"""
    offsets(samples, i)

The `2 × nsamps` matrix of offset draws for footprint `i`, reconstructed from
the stored grid indices.  Useful when `store_h = false` keeps the full
coordinate array out of memory.
"""
function offsets(samples, i::Integer)
    g = samples.grid
    idx = view(samples.hidx, i, :)
    out = Matrix{Float64}(undef, 2, length(idx))
    @inbounds for m in eachindex(idx)
        out[1, m] = g.coords[idx[m], 1]
        out[2, m] = g.coords[idx[m], 2]
    end
    return out
end

"""
    posteriormean(samples)

`n × 2` posterior mean offset per footprint, computed from the stored indices.
"""
function posteriormean(samples)
    g = samples.grid
    idx = samples.hidx
    n, M = size(idx)
    out = zeros(Float64, n, 2)
    @inbounds for m in 1:M, i in 1:n
        out[i, 1] += g.coords[idx[i, m], 1]
        out[i, 2] += g.coords[idx[i, m], 2]
    end
    return out ./ M
end
