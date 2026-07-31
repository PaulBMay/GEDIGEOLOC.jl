# ---------------------------------------------------------------------------
# Prediction at held-out footprints.
#
# For an AR(1) the conditional of w at an interior point given its two flanking
# footprints is the Brownian-bridge analogue, available in closed form — no
# neighbour sets, no local Cholesky, no k-d tree.  With
#
#   a = exp(-(tp - t_j)/ϕ),  b = exp(-(t_{j+1} - tp)/ϕ),  c = ab
#
#   E[w_p | w_j, w_{j+1}] = [a(1-b²)w_j + b(1-a²)w_{j+1}] / (1-c²)
#   Var[w_p | ·]          = (1-a²)(1-b²)/(1-c²)   (correlation scale)
#
# and outside the observed range it collapses to the one-sided AR(1) step.
# ---------------------------------------------------------------------------

"""
    predoffsets(samples, tp, t, grid; every = 1)

Posterior predictive offsets at the coordinates `tp`, given the draws returned
by [`geolocate`](@ref) for footprints at `t`.

Returns `(offsets, gridindex)`: an `np × 2 × nsamps` array of offset draws and
the `np × nsamps` index of the nearest grid cell to each.

`every` thins the input draws.  `samples` must have been produced with
`store_w = true`.
"""
function predoffsets(samples, tp::AbstractVecOrMat, t::AbstractVecOrMat, grid; every::Integer = 1)
    samples.w === nothing && throw(ArgumentError(
        "predoffsets needs the latent field; rerun geolocate with store_w = true"))

    g = OffsetGrid(grid)
    tv = Float64.(vec(t))
    tpv = Float64.(vec(tp))
    n, np = length(tv), length(tpv)

    idx = 1:every:length(samples.ϕ)
    M = length(idx)

    # Bracketing footprints and the lags to them: fixed across draws.
    left = Vector{Int}(undef, np)
    dl = Vector{Float64}(undef, np)
    dr = Vector{Float64}(undef, np)
    @inbounds for p in 1:np
        j = searchsortedlast(tv, tpv[p])
        if j < 1
            left[p] = 0
            dl[p] = 0.0
            dr[p] = tv[1] - tpv[p]
        elseif j ≥ n
            left[p] = n
            dl[p] = tpv[p] - tv[n]
            dr[p] = 0.0
        else
            left[p] = j
            dl[p] = tpv[p] - tv[j]
            dr[p] = tv[j+1] - tpv[p]
        end
    end

    out = Array{Float64}(undef, np, 2, M)
    gidx = Matrix{Int32}(undef, np, M)

    for (m, mm) in enumerate(idx)
        ϕ = samples.ϕ[mm]
        ρ = samples.ρ[mm]
        μ1, μ2 = samples.μ[1, mm], samples.μ[2, mm]
        Σ = view(samples.Σ, :, :, mm)
        U = cholesky(Symmetric(Matrix(Σ))).U
        u11, u12, u22 = U[1, 1], U[1, 2], U[2, 2]
        W = view(samples.w, :, :, mm)

        @inbounds for p in 1:np
            j = left[p]
            if j == 0                                  # before the first footprint
                a = exp(-dr[p] / ϕ)
                mw1, mw2 = a * W[1, 1], a * W[1, 2]
                v = 1 - a * a
            elseif j == n                              # after the last
                a = exp(-dl[p] / ϕ)
                mw1, mw2 = a * W[n, 1], a * W[n, 2]
                v = 1 - a * a
            else                                       # AR(1) bridge
                a = exp(-dl[p] / ϕ)
                b = exp(-dr[p] / ϕ)
                c2 = (a * b)^2
                den = 1 - c2
                ca = a * (1 - b * b) / den
                cb = b * (1 - a * a) / den
                mw1 = ca * W[j, 1] + cb * W[j+1, 1]
                mw2 = ca * W[j, 2] + cb * W[j+1, 2]
                v = (1 - a * a) * (1 - b * b) / den
            end
            sd = sqrt(ρ * v + (1 - ρ))                 # bridge + nugget
            z1, z2 = sd * randn(), sd * randn()
            x = μ1 + mw1 + z1 * u11
            yv = μ2 + mw2 + z1 * u12 + z2 * u22
            out[p, 1, m] = x
            out[p, 2, m] = yv
            gidx[p, m] = nearestindex(g, x, yv)
        end
    end

    return out, gidx
end
