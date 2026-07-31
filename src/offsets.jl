# ---------------------------------------------------------------------------
# The offset (h) update — the inner loop of the whole sampler.
#
# For each footprint i we draw a grid cell j from
#
#   log p(h_i = g_j | ·) ∝ -½ τ_i (R[j,i] + α - y_i)²  -  ½ sᵢ (g_j - mᵢ)ᵀ A₀ (g_j - mᵢ)
#
# Expanding the prior quadratic and dropping the term that is constant in j,
#
#   log p ∝ -½ τ_i (R[j,i] + α - y_i)²  +  sᵢ (qA[j] + g_j · vᵢ)
#
# with qA[j] = -½ g_jᵀA₀g_j precomputed once per sweep and vᵢ = A₀mᵢ.  The
# per-(i,j) cost is then a few FMAs over one streaming read of R[j,i], with no
# k-length temporaries, no `Categorical` construction, and no separate
# normalisation pass.
# ---------------------------------------------------------------------------

"""
    OffsetWorkspace(k, n; nchunks, rng)

Per-chunk scratch and RNGs for the offset update.  One length-`k` buffer per
chunk stays resident in cache; nothing else is allocated during sampling.
"""
struct OffsetWorkspace
    qA::Vector{Float64}
    buf::Vector{Vector{Float64}}
    blk::Vector{Vector{Float64}}    # per-block partial sums, see `_pick`
    rng::Vector{Xoshiro}
    ranges::Vector{UnitRange{Int}}
    V::Matrix{Float64}          # n × 2, vᵢ = A₀mᵢ
    τobs::Vector{Float64}       # n, observation precision per footprint
end

function OffsetWorkspace(k::Integer, n::Integer; nchunks::Integer = 1,
                         rng::AbstractRNG = Random.default_rng())
    ranges = _chunkranges(n, nchunks)
    nc = length(ranges)
    nblk = cld(k, _BLK)
    return OffsetWorkspace(zeros(Float64, k),
                           [zeros(Float64, k) for _ in 1:nc],
                           [zeros(Float64, nblk) for _ in 1:nc],
                           [Xoshiro(rand(rng, UInt64)) for _ in 1:nc],
                           ranges, zeros(Float64, n, 2), zeros(Float64, n))
end

function _chunkranges(n::Integer, nchunks::Integer)
    nc = clamp(nchunks, 1, max(n, 1))
    return [(((c - 1) * n) ÷ nc + 1):((c * n) ÷ nc) for c in 1:nc]
end

_windowable(g::OffsetGrid) = g.lattice && g.dx > 0 && g.dy > 0

# --- vectorisable exp ------------------------------------------------------

const _LOG2E = 1.4426950408889634
const _EXPMAGIC = 6755399441055744.0    # 2^52 + 2^51

"""
    expfast(x)

`exp(x)` for `x ≤ 0`, accurate to a relative error of `3e-10` and — unlike
`Base.exp` — built only from arithmetic, so LLVM vectorises it.

This matters because normalising the categorical draw needs one exponential per
grid cell per footprint: `Base.exp` costs ~4 ns/element against ~0.3 ns for the
rest of the inner loop combined, and so dominates the whole sampler.  A relative
error of `3e-10` on a probability that is then normalised is many orders of
magnitude below the Monte Carlo error of any feasible chain.

Writes `x` as `n + f` in units of `log2(e)` with `f ∈ [-½, ½]`; `2^n` is built
by adding to the exponent field and `2^f` from a degree-8 minimax polynomial.
Adding `2^52 + 2^51` performs the rounding and extracts `n` in one step, with no
float→int conversion.  Inputs below `-700` are clamped, which keeps the exponent
field in range and is exact to within `1e-304`.
"""
@inline function expfast(x::Float64)
    t = max(x, -700.0) * _LOG2E
    u = t + _EXPMAGIC
    f = t - (u - _EXPMAGIC)
    scale = reinterpret(Float64, (reinterpret(UInt64, u) + 0x3ff) << 52)
    p = evalpoly(f, (1.0, 0.6931471805599453, 0.2402265069591007,
                     0.05550410866482158, 0.009618129107628477,
                     0.0013333558146428443, 1.5403530393381609e-4,
                     1.5252733804059840e-5, 1.3215486790144306e-6))
    return scale * p
end

"""
    fill_qA!(qA, grid, A₀)

`qA[j] = -½ g_jᵀ A₀ g_j`.  Depends only on `Σ` and `ρ`, so it is recomputed
once per sweep rather than once per footprint.
"""
function fill_qA!(qA::Vector{Float64}, g::OffsetGrid, A0::NTuple{3,Float64})
    a11, a12, a22 = A0
    C = g.coords
    @inbounds @simd for j in eachindex(qA)
        x, y = Float64(C[j, 1]), Float64(C[j, 2])
        qA[j] = -0.5 * (a11 * x * x + 2 * a12 * x * y + a22 * y * y)
    end
    return qA
end

# --- normalise-and-draw ----------------------------------------------------

const _BLK = 64

"""
    _pick(buf, blk, m, mx, rng)

Exponentiate `buf[1:m] .- mx`, then draw an index from the resulting weights.
Returns the index.

The naive version — one running cumulative sum until it passes `u` — is a
*serial* floating-point dependency chain averaging `m/2` additions, which at
`m = 2601` was costing more than the likelihood evaluation itself.  Instead the
exponential pass accumulates a partial sum per block of `_BLK`, so the draw
scans at most `m/_BLK + _BLK` links serially (≈ 100 rather than ≈ 1300) while
the arithmetic inside each block still vectorises.
"""
@inline function _pick(buf::Vector{Float64}, blk::Vector{Float64}, m::Int,
                       mx::Float64, rng::AbstractRNG)
    nb = cld(m, _BLK)
    tot = 0.0
    @inbounds for b in 1:nb
        lo = (b - 1) * _BLK + 1
        hi = min(b * _BLK, m)
        s = 0.0
        @simd for j in lo:hi
            e = expfast(buf[j] - mx)
            buf[j] = e
            s += e
        end
        blk[b] = s
        tot += s
    end

    u = rand(rng) * tot
    acc = 0.0
    b = 1
    @inbounds while b < nb && acc + blk[b] < u
        acc += blk[b]
        b += 1
    end
    lo = (b - 1) * _BLK + 1
    hi = min(b * _BLK, m)
    @inbounds for j in lo:hi
        acc += buf[j]
        acc ≥ u && return j
    end
    return hi
end

# --- inner draws -----------------------------------------------------------

"""
Full scan over every grid cell.  Exact: the only pass that touches the big
reference matrix is the first.
"""
@inline function _draw_full(buf::Vector{Float64}, blk::Vector{Float64},
                            R::AbstractMatrix, i::Int,
                            gx, gy, qA::Vector{Float64},
                            v1::Float64, v2::Float64, s::Float64,
                            c::Float64, τ::Float64, rng::AbstractRNG)
    k = length(qA)
    mx = -Inf
    @inbounds @simd for j in 1:k
        t = Float64(R[j, i]) + c
        lp = muladd(s, muladd(gy[j], v2, muladd(gx[j], v1, qA[j])), -0.5 * τ * t * t)
        buf[j] = lp
        mx = max(mx, lp)
    end
    return _pick(buf, blk, k, mx, rng)
end

"""
Scan only the lattice cells within `hw` standard deviations of the prior mean.
Returns 0 when the window is empty so the caller can fall back to a full scan.
"""
@inline function _draw_window(buf::Vector{Float64}, blk::Vector{Float64},
                              R::AbstractMatrix, i::Int,
                              g::OffsetGrid, qA::Vector{Float64},
                              v1::Float64, v2::Float64, s::Float64,
                              c::Float64, τ::Float64,
                              m1::Float64, m2::Float64, hw1::Float64, hw2::Float64,
                              rng::AbstractRNG)
    ny = g.ny
    alo = max(1, ceil(Int, (m1 - hw1 - g.x0) / g.dx) + 1)
    ahi = min(g.nx, floor(Int, (m1 + hw1 - g.x0) / g.dx) + 1)
    blo = max(1, ceil(Int, (m2 - hw2 - g.y0) / g.dy) + 1)
    bhi = min(ny, floor(Int, (m2 + hw2 - g.y0) / g.dy) + 1)
    (alo > ahi || blo > bhi) && return 0

    nb = bhi - blo + 1
    mx = -Inf
    p = 0
    @inbounds for a in alo:ahi
        xv1 = (g.x0 + (a - 1) * g.dx) * v1
        base = (a - 1) * ny
        @simd for b in blo:bhi
            yv2 = (g.y0 + (b - 1) * g.dy) * v2
            t = Float64(R[base+b, i]) + c
            lp = muladd(s, qA[base+b] + xv1 + yv2, -0.5 * τ * t * t)
            buf[p+b-blo+1] = lp
            mx = max(mx, lp)
        end
        p += nb
    end

    win = _pick(buf, blk, p, mx, rng)
    ca, qb = divrem(win - 1, nb)
    return (alo + ca - 1) * ny + blo + qb
end

# --- sweeps ----------------------------------------------------------------

"""
Nugget model: given `w` the `h_i` are conditionally independent, so the whole
sweep is one block update and parallelises over contiguous footprint chunks.
"""
function _sweep_indep!(rows::UnitRange{Int}, hidx, h, curref, g::OffsetGrid,
                       R::AbstractMatrix, y, α::Float64, τobs, V, qA,
                       buf, blk, rng, μ1::Float64, μ2::Float64, w,
                       hw1::Float64, hw2::Float64, usewindow::Bool)
    C = g.coords
    gx, gy = view(C, :, 1), view(C, :, 2)
    @inbounds for i in rows
        v1, v2 = V[i, 1], V[i, 2]
        c = α - y[i]
        τ = τobs[i]
        j = 0
        if usewindow
            j = _draw_window(buf, blk, R, i, g, qA, v1, v2, 1.0, c, τ,
                             μ1 + w[i, 1], μ2 + w[i, 2], hw1, hw2, rng)
        end
        j == 0 && (j = _draw_full(buf, blk, R, i, gx, gy, qA, v1, v2, 1.0, c, τ, rng))
        hidx[i] = j
        h[i, 1] = C[j, 1]
        h[i, 2] = C[j, 2]
        curref[i] = R[j, i]
    end
    return nothing
end

"""
Nugget-free model: `h` itself is the AR(1), so this is a sequential single-site
Gibbs sweep conditioning on the two adjacent footprints through the tridiagonal
precision.
"""
function _sweep_markov!(hidx, h, curref, g::OffsetGrid, R::AbstractMatrix, y,
                        α::Float64, τobs, μ, ar::AR1, A0::NTuple{3,Float64},
                        qA, buf, blk, rng, window::Float64, V11::Float64, V22::Float64)
    C = g.coords
    gx, gy = view(C, :, 1), view(C, :, 2)
    a11, a12, a22 = A0
    dv, ev = ar.Kinv.dv, ar.Kinv.ev
    n = length(dv)
    μ1, μ2 = Float64(μ[1]), Float64(μ[2])
    usewindow = isfinite(window) && _windowable(g)

    @inbounds for i in 1:n
        s = dv[i]
        nb1 = nb2 = 0.0
        if i > 1
            nb1 += ev[i-1] * (h[i-1, 1] - μ1)
            nb2 += ev[i-1] * (h[i-1, 2] - μ2)
        end
        if i < n
            nb1 += ev[i] * (h[i+1, 1] - μ1)
            nb2 += ev[i] * (h[i+1, 2] - μ2)
        end
        m1 = μ1 - nb1 / s
        m2 = μ2 - nb2 / s
        v1 = a11 * m1 + a12 * m2
        v2 = a12 * m1 + a22 * m2
        c = α - y[i]
        τ = τobs[i]
        j = 0
        if usewindow
            j = _draw_window(buf, blk, R, i, g, qA, v1, v2, s, c, τ, m1, m2,
                             window * sqrt(V11 / s), window * sqrt(V22 / s), rng)
        end
        j == 0 && (j = _draw_full(buf, blk, R, i, gx, gy, qA, v1, v2, s, c, τ, rng))
        hidx[i] = j
        h[i, 1] = C[j, 1]
        h[i, 2] = C[j, 2]
        curref[i] = R[j, i]
    end
    return nothing
end

"""
    sample_offsets!(hidx, h, curref, ws, grid, reference, y, α, μ, w, Σ, ar; window)

Draw every `h_i` from its full conditional.  Dispatches on whether the offset
prior has a nugget: with one, the draws are conditionally independent given `w`
and run in parallel; without one they form a sequential sweep along the chain.
"""
function sample_offsets!(hidx, h, curref, ws::OffsetWorkspace, g::OffsetGrid,
                         R::AbstractMatrix, y, α::Float64, μ, w, Σ,
                         ar::AR1; window::Float64 = Inf)
    # Prior covariance shared by all footprints, up to the per-footprint scale sᵢ.
    P = hasnugget(ar) ? (1 - ar.ρ) .* Σ : Σ
    dt = P[1, 1] * P[2, 2] - P[1, 2] * P[2, 1]
    A0 = (P[2, 2] / dt, -P[1, 2] / dt, P[1, 1] / dt)
    fill_qA!(ws.qA, g, A0)

    if !hasnugget(ar)
        _sweep_markov!(hidx, h, curref, g, R, y, α, ws.τobs, μ, ar, A0,
                       ws.qA, ws.buf[1], ws.blk[1], ws.rng[1], window, P[1, 1], P[2, 2])
        return nothing
    end

    a11, a12, a22 = A0
    μ1, μ2 = Float64(μ[1]), Float64(μ[2])
    V = ws.V
    @inbounds @simd for i in axes(V, 1)
        m1 = μ1 + w[i, 1]
        m2 = μ2 + w[i, 2]
        V[i, 1] = a11 * m1 + a12 * m2
        V[i, 2] = a12 * m1 + a22 * m2
    end

    usewindow = isfinite(window) && _windowable(g)
    hw1 = window * sqrt(P[1, 1])
    hw2 = window * sqrt(P[2, 2])
    ranges, bufs, blks, rngs = ws.ranges, ws.buf, ws.blk, ws.rng

    if length(ranges) == 1
        _sweep_indep!(ranges[1], hidx, h, curref, g, R, y, α, ws.τobs, V, ws.qA,
                      bufs[1], blks[1], rngs[1], μ1, μ2, w, hw1, hw2, usewindow)
    else
        @sync for c in eachindex(ranges)
            Threads.@spawn _sweep_indep!(ranges[c], hidx, h, curref, g, R, y, α,
                                         ws.τobs, V, ws.qA, bufs[c], blks[c], rngs[c],
                                         μ1, μ2, w, hw1, hw2, usewindow)
        end
    end
    return nothing
end
