# GEDIGEOLOC

[![Build Status](https://github.com/PaulBMay/GEDIGEOLOC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/PaulBMay/GEDIGEOLOC.jl/actions/workflows/CI.yml?query=branch%3Amain)

Bayesian inference for GEDI geolocation offsets within a contiguous beam
collection.

Each footprint's true ground position is displaced from its reported position by
an unknown 2-D offset. The offsets are inferred by scoring candidate offsets
against a reference DEM, under a prior that ties neighbouring footprints
together along track.

## Model

For footprint `i` at along-track coordinate `t[i]`, with candidate offsets
`grid` (`k × 2`) and reference values `reference[j, i]` at offset `grid[j, :]`:

```
y_i    = α + reference[h_i, i] + e_i           e_i ~ N(0, 1/τ)
h_i    = μ + w_i + ε_i                         ε_i ~ N(0, (1-ρ)Σ)
cor(w_i, w_j) = exp(-|t_i - t_j| / ϕ)          Var(w_i) = ρΣ
```

`h_i` is constrained to the grid. `ρ ∈ (0,1]` splits the offset variance between
a smooth along-track component `w` and per-shot jitter `ε`.

Two variants, both keyword-selected:

* `mixture = true` — `e_i` is a two-component normal mixture with weight `πz` on
  the inflated component, absorbing ground-finding errors.
* `nugget = false` — `ρ ≡ 1`, so `h` is itself the AR(1) chain.

## Quick start

```julia
using GEDIGEOLOC, LinearAlgebra

grid = OffsetGrid(-25.0:1.0:25.0, -25.0:1.0:25.0)   # k × 2 candidate offsets
# reference :: k × n   reference (DEM) elevations
# y :: n               observed elevations
# t :: n               strictly increasing along-track coordinate

priors = (α = (0.0, 100.0),                  # (mean, precision)
          τ = gammashaperate(100.0, 10.0),   # (shape, rate)
          Σ = (50.0, diagm(fill(100.0, 2))), # (ν₀, Ψ)
          ϕ = gammashaperate(5_000.0, 1_000.0),
          ρ = logitmeansd(0.90, 0.98, 0.99)) # prior on the logit scale

initial = (w = zeros(n, 2), Σ = diagm(fill(100.0, 2)), α = 0.0,
           ϕ = 5_000.0, ρ = 0.95, μ = [0.0, 0.0], τ = 100.0)

samples = geolocate(y, t, grid, reference, priors, initial, 2_000; burn = 500)

posteriormean(samples)      # n × 2 posterior mean offset
samples.accept              # Metropolis acceptance rate for (ϕ, ρ)
```

`demonstration/geoloc_demo.jl` runs the whole thing end to end on simulated
data, including prediction at held-out footprints via `predoffsets`.

## Why it is fast

A Matérn ν = 1/2 field in 1-D is Markov, so `K⁻¹` is tridiagonal. Factorising
it, solving with it, taking its log-determinant and drawing `w` are all `O(n)`
with no allocation — no nearest-neighbour sets, no sparse triple product, no
fill-reducing permutation.

That leaves the categorical draw over the offset grid as the entire cost. For
each footprint it evaluates

```
log p(h_i = g_j | ·) ∝ -½τ_i(R[j,i] + α - y_i)² + s_i(qA[j] + g_j·v_i)
```

in one streaming pass over `R[:, i]`, with the prior quadratic pre-expanded so
that only a few FMAs per cell remain. Normalising and drawing then use a
vectorisable `exp` and a two-level cumulative scan, both of which matter more
than they sound like they should — see the comments in `src/offsets.jl`.

Measured on a Threadripper 7985WX at `n = 4000`, `k = 2601`, per Gibbs sweep:

| | ms/sweep | alloc/sweep |
|---|---|---|
| previous implementation | 117.2 | 505 MiB |
| current, 1 thread | 11.1 | 0.28 MiB |
| current, 16 threads | 1.35 | 0.32 MiB |

Three optional knobs trade nothing, memory, or a little accuracy for more speed:

* `nchunks` — parallelise the offset draw over footprints. Exact: with a nugget
  the `h_i` are conditionally independent given `w`, so the sweep is one block
  update.
* `reference::Matrix{Float32}` — halves the memory traffic of the dominant loop.
  Worth ~25% once `k × n` exceeds cache.
* `window = 6.0` — restrict each draw to the grid cells within 6 prior standard
  deviations of its prior mean. **This is an approximation**: the dropped cells
  have negligible prior mass, but a sharp reference likelihood can favour a far
  cell, so a jump the exact sampler would make can be suppressed. Check against
  `window = Inf` before relying on it. Pays off on large grids (5× at
  `k = 10201`) and costs a little on small ones.

## Notes on the sampler

The sweep order is `h → α → τ → (z, πz) → Σ → μ → (ϕ, ρ) → w`.

The `Σ`, `μ` and `(ϕ, ρ)` updates all target densities with `w` integrated out,
so `w` must be redrawn after them and before it is next used. Drawing `w` before
the `(ϕ, ρ)` update — as the pre-refactor code did — leaves the chain with the
wrong stationary distribution.

The `(ϕ, ρ)` Metropolis step adapts its step size by Robbins–Monro during
burn-in, targeting a 30% acceptance rate.

`geolocate2` and `geolocate_gfe` are deprecated shims for
`geolocate(...; nugget = false)` and `geolocate(...; mixture = true)`.
