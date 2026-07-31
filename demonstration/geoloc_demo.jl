# Simulate a beam collection, fit the three model variants, and predict at
# held-out footprints.  Run with `julia --project=. -t auto`.

using GEDIGEOLOC
using LinearAlgebra, Random, Distributions, Statistics

Random.seed!(96)

# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------

n = 2_000
spacing = 60.0                       # metres between footprints
t = spacing .* (1:n)                 # along-track coordinate

grid = OffsetGrid(-25.0:1.0:25.0, -25.0:1.0:25.0)
k = length(grid)

# Reference (DEM) values: a smooth surface over the offset grid, per footprint.
Σgrid = GEDIGEOLOC.expcor(grid.coords, 12.0) + 1e-8I
reference = cholesky(Symmetric(Σgrid)).L * randn(k, n)

# True offsets: AR(1) along track plus a nugget.
ϕtrue, ρtrue = 5_000.0, 0.95
Σtrue = 50.0 * [1.0 0.7; 0.7 1.0]
μtrue = [4.0, -6.0]
Ctrue = ρtrue .* [exp(-abs(a - b) / ϕtrue) for a in t, b in t] + (1 - ρtrue) * I(n)
offsets_true = μtrue' .+ cholesky(Symmetric(Ctrue)).L * randn(n, 2) * cholesky(Σtrue).U

# Snap to the grid, then observe with noise plus occasional ground-finding error.
idx = [nearestindex(grid, offsets_true[i, 1], offsets_true[i, 2]) for i in 1:n]
snapped = grid.coords[idx, :]
τtrue = [100.0, 1.0]
ztrue = rand(n) .< 0.05
y = [reference[idx[i], i] for i in 1:n] .+ sqrt.(1 ./ τtrue[ztrue .+ 1]) .* randn(n)

rmse(a, b) = sqrt(mean(abs2, a .- b))

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

priors = (α  = (0.0, 100.0),
          τ  = gammashaperate(100.0, 10.0),
          Σ  = (50.0, diagm(fill(100.0, 2))),
          ϕ  = gammashaperate(5_000.0, 1_000.0),
          ρ  = logitmeansd(0.90, 0.98, 0.99))      # prior on the logit scale

initial = (w = zeros(n, 2), Σ = diagm(fill(100.0, 2)), α = 0.0,
           ϕ = 5_000.0, ρ = 0.95, μ = [0.0, 0.0], τ = 100.0)

samples = geolocate(y, t, grid, reference, priors, initial, 2_000; burn = 500)

@show samples.accept
@show rmse(posteriormean(samples), snapped)
@show mean(samples.ϕ), mean(samples.ρ)
@show mean(samples.μ, dims = 2)

# Two-component error, to absorb the ground-finding failures.
priors_mix = (α  = (0.0, 100.0),
              τ₀ = gammashaperate(100.0, 10.0),
              τ₁ = gammashaperate(1.0, 0.5),
              πz = betaparams(0.05, 0.05),
              Σ  = (50.0, diagm(fill(100.0, 2))),
              ϕ  = gammashaperate(5_000.0, 1_000.0),
              ρ  = logitmeansd(0.90, 0.98, 0.99))
initial_mix = merge(initial, (τ = [100.0, 1.0], πz = 0.05, z = ztrue))

samples_mix = geolocate(y, t, grid, reference, priors_mix, initial_mix, 2_000;
                        mixture = true, burn = 500)

@show rmse(posteriormean(samples_mix), snapped)
@show mean(samples_mix.z[ztrue, :]), mean(samples_mix.z[.!ztrue, :])

# Nugget-free: h is itself the AR(1) chain.
samples_nn = geolocate(y, t, grid, reference, priors, initial, 2_000;
                       nugget = false, burn = 500)
@show rmse(posteriormean(samples_nn), snapped)

# ---------------------------------------------------------------------------
# Speed knobs
# ---------------------------------------------------------------------------

# `reference` may be Float32 (halves the memory traffic of the dominant loop),
# `nchunks` parallelises over footprints, and `window` restricts each draw to the
# grid cells the prior supports.  `window` is an approximation — check it against
# `window = Inf` before relying on it.
fast = geolocate(y, t, grid, Float32.(reference), priors, initial, 2_000;
                 burn = 500, nchunks = Threads.nthreads(), window = 6.0)
@show rmse(posteriormean(fast), snapped)

# Skip the big coordinate arrays; `hidx` is always returned and `posteriormean`
# reconstructs from it.
lean = geolocate(y, t, grid, reference, priors, initial, 2_000;
                 burn = 500, store_h = false, store_w = false)
@show rmse(posteriormean(lean), snapped)

# ---------------------------------------------------------------------------
# Predict at held-out footprints
# ---------------------------------------------------------------------------

hold = collect(200:2:400)
train = setdiff(1:n, hold)

initial_train = merge(initial, (w = zeros(length(train), 2),))
s_train = geolocate(y[train], t[train], grid, reference[:, train],
                    priors, initial_train, 2_000; burn = 500)

pred, predidx = predoffsets(s_train, t[hold], t[train], grid; every = 10)
pred_mean = dropdims(mean(pred, dims = 3), dims = 3)

@show rmse(pred_mean, offsets_true[hold, :])
@show cor(vec(pred_mean), vec(offsets_true[hold, :]))

# `predidx` gives the grid cell each predictive draw lands in, which is what you
# want for propagating the offset uncertainty into the reference lookup.
@show size(predidx)
