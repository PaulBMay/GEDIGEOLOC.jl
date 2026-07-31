using GEDIGEOLOC
using Test
using LinearAlgebra, Random, Statistics, Distributions

const G = GEDIGEOLOC

# Dense reference implementations to check the O(n) machinery against.
densecor(t, ϕ) = [exp(-abs(ti - tj) / ϕ) for ti in t, tj in t]
densecov(t, ϕ, ρ) = ρ * densecor(t, ϕ) + (1 - ρ) * I(length(t))

@testset "GEDIGEOLOC.jl" begin

    @testset "TriLDL" begin
        Random.seed!(1)
        n = 40
        Q = SymTridiagonal(rand(n) .+ 4, randn(n - 1))
        D = Matrix(Q)
        F = G.ldl!(G.TriLDL{Float64}(n), Q)

        @test logdet(F) ≈ logdet(D)

        b = randn(n)
        @test ldiv!(F, copy(b)) ≈ D \ b
        B = randn(n, 2)
        @test ldiv!(F, copy(B)) ≈ D \ B

        # L⁻ᵀD^{-1/2} maps iid normals to N(0, Q⁻¹)
        Z = Matrix(1.0I, n, n)
        A = G.whitensolve!(F, Z)
        @test A * A' ≈ inv(D)
    end

    @testset "AR1 against dense" begin
        Random.seed!(2)
        n = 30
        t = cumsum(rand(n) .+ 0.5)          # irregular lags
        for ρ in (0.85, 1.0), ϕ in (0.7, 3.0)
            ar = G.AR1(t; phi = ϕ, rho = ρ)
            C = densecov(t, ϕ, ρ)

            @test ar.logdetC ≈ logdet(C)
            @test ar.logdetK ≈ logdet(densecor(t, ϕ))

            X = randn(n, 2)
            @test G.applyCinv!(copy(X), ar) ≈ C \ X
            @test ar.onesolve ≈ C \ ones(n)
            @test ar.onesum ≈ sum(C \ ones(n))

            # update! must fully refresh state, whatever it held before
            ar2 = G.AR1(t; phi = 11.0, rho = 0.5)
            G.update!(ar2, ϕ, ρ)
            @test ar2.logdetC ≈ ar.logdetC
            @test G.applyCinv!(copy(X), ar2) ≈ C \ X
        end
    end

    @testset "sample_w! moments" begin
        # This is the step whose sparse-Cholesky predecessor permuted with `F.p`
        # instead of `invperm(F.p)`, giving w a scrambled covariance.
        Random.seed!(3)
        n, ϕ, ρ = 12, 2.0, 0.8
        t = collect(1.0:n)
        ar = G.AR1(t; phi = ϕ, rho = ρ)
        Σ = [2.0 0.6; 0.6 1.5]
        ΣU = cholesky(Symmetric(Σ)).U
        res = randn(n, 2)

        K = densecor(t, ϕ)
        # w | h has precision (ρK)⁻¹ ⊗ Σ⁻¹ + (1-ρ)⁻¹I ⊗ Σ⁻¹
        Qw = inv(ρ * K) + I(n) / (1 - ρ)
        Cw = inv(Qw)
        Mw = Cw * res / (1 - ρ)

        M = 200_000
        w = zeros(n, 2)
        Z = zeros(n, 2)
        acc = zeros(n, 2)
        S11 = zeros(n, n)
        for _ in 1:M
            G.sample_w!(w, ar, res, ΣU, Z, Random.default_rng())
            acc .+= w
            d = w[:, 1] .- Mw[:, 1]
            S11 .+= d * d'
        end
        @test acc ./ M ≈ Mw rtol = 0.05 atol = 0.02
        @test S11 ./ M ≈ Σ[1, 1] .* Cw rtol = 0.08 atol = 0.05

        # ρ = 1: no nugget, w is determined by h
        arnn = G.AR1(t; phi = ϕ, rho = 1.0)
        @test G.sample_w!(w, arnn, res, ΣU, Z, Random.default_rng()) == res
    end

    @testset "OffsetGrid" begin
        x, y = -2.0:1.0:2.0, -3.0:1.5:3.0
        g = OffsetGrid(x, y)
        @test islattice(g)
        @test length(g) == length(x) * length(y)

        # round-trip through a plain matrix, which is how callers build grids
        g2 = OffsetGrid(g.coords)
        @test islattice(g2)
        @test (g2.x0, g2.dx, g2.nx) == (g.x0, g.dx, g.nx)
        @test (g2.y0, g2.dy, g2.ny) == (g.y0, g.dy, g.ny)

        @test g.coords[centerindex(g), :] == [0.0, 0.0]
        for _ in 1:200
            px, py = 6randn(), 6randn()
            brute = argmin(vec(sum((g.coords .- [px py]) .^ 2, dims = 2)))
            @test nearestindex(g, px, py) == brute
        end

        scattered = OffsetGrid(randn(17, 2))
        @test !islattice(scattered)
        px, py = 0.3, -0.2
        @test nearestindex(scattered, px, py) ==
              argmin(vec(sum((scattered.coords .- [px py]) .^ 2, dims = 2)))
    end

    @testset "offset draws match the exact conditional" begin
        Random.seed!(4)
        n = 6
        g = OffsetGrid(-1.0:1.0:1.0, -1.0:1.0:1.0)
        k = length(g)
        R = randn(k, n)
        y = randn(n)
        α, τ = 0.3, 4.0
        μ = [0.2, -0.4]
        Σ = [1.3 0.4; 0.4 0.9]
        ρ, ϕ = 0.7, 5.0
        ar = G.AR1(collect(1.0:n); phi = ϕ, rho = ρ)
        w = 0.5 .* randn(n, 2)

        P = (1 - ρ) * Σ
        function exactprobs(i)
            lp = [(-0.5τ * (R[j, i] + α - y[i])^2 +
                   logpdf(MvNormal(μ + w[i, :], Symmetric(P)), g.coords[j, :]))
                  for j in 1:k]
            e = exp.(lp .- maximum(lp))
            return e ./ sum(e)
        end

        ws = G.OffsetWorkspace(k, n; nchunks = 1, rng = Random.default_rng())
        fill!(ws.τobs, τ)
        hidx = zeros(Int32, n)
        h = zeros(n, 2)
        curref = zeros(n)

        M = 60_000
        counts = zeros(Int, k, n)
        for _ in 1:M
            G.sample_offsets!(hidx, h, curref, ws, g, R, y, α, μ, w, Σ, ar)
            for i in 1:n
                counts[hidx[i], i] += 1
            end
        end
        for i in 1:n
            @test counts[:, i] ./ M ≈ exactprobs(i) atol = 0.01
        end

        # h and curref must stay consistent with the drawn index
        for i in 1:n
            @test h[i, :] == g.coords[hidx[i], :]
            @test curref[i] == R[hidx[i], i]
        end
    end

    @testset "nugget-free sweep targets the exact joint" begin
        # Small enough to enumerate every configuration of the chain.
        Random.seed!(5)
        n = 3
        g = OffsetGrid(-0.5:1.0:0.5, -0.5:1.0:0.5)   # k = 4
        k = length(g)
        R = randn(k, n)
        y = randn(n)
        α, τ = 0.0, 2.0
        μ = [0.1, 0.0]
        Σ = [1.0 0.3; 0.3 1.2]
        t = collect(1.0:n)
        ϕ = 2.0
        ar = G.AR1(t; phi = ϕ, rho = 1.0)
        C = densecor(t, ϕ)

        cfgs = collect(Iterators.product(ntuple(_ -> 1:k, n)...))
        logw = map(cfgs) do c
            H = vcat((g.coords[c[i], :]' for i in 1:n)...)
            ll = sum(-0.5τ * (R[c[i], i] + α - y[i])^2 for i in 1:n)
            res = H .- μ'
            lp = -0.5 * (tr(inv(Σ) * res' * (C \ res)) + n * logdet(Σ) + 2 * logdet(C))
            ll + lp
        end
        pj = exp.(logw .- maximum(logw))
        pj ./= sum(pj)
        exactmarg = [sum(pj[[c[i] == j for c in cfgs]]) for j in 1:k, i in 1:n]

        ws = G.OffsetWorkspace(k, n; nchunks = 1, rng = Random.default_rng())
        fill!(ws.τobs, τ)
        hidx = fill(Int32(1), n)
        h = repeat(g.coords[1:1, :], n)
        curref = zeros(n)
        w = zeros(n, 2)

        M = 200_000
        counts = zeros(Int, k, n)
        for _ in 1:M
            G.sample_offsets!(hidx, h, curref, ws, g, R, y, α, μ, w, Σ, ar)
            for i in 1:n
                counts[hidx[i], i] += 1
            end
        end
        @test counts ./ M ≈ exactmarg atol = 0.01
    end

    @testset "window approximation" begin
        Random.seed!(6)
        n = 40
        g = OffsetGrid(-10.0:1.0:10.0, -10.0:1.0:10.0)
        k = length(g)
        R = 0.05 .* randn(k, n)          # weak likelihood: the prior dominates
        y = zeros(n)
        μ = [0.0, 0.0]
        Σ = [1.0 0.0; 0.0 1.0]
        ar = G.AR1(collect(1.0:n); phi = 5.0, rho = 0.5)
        w = zeros(n, 2)
        hidx = zeros(Int32, n); h = zeros(n, 2); curref = zeros(n)

        function tally(window, seed)
            ws = G.OffsetWorkspace(k, n; nchunks = 1, rng = Xoshiro(seed))
            fill!(ws.τobs, 1.0)
            c = zeros(Int, k)
            for _ in 1:4_000
                G.sample_offsets!(hidx, h, curref, ws, g, R, y, 0.0, μ, w, Σ, ar;
                                  window = window)
                for i in 1:n
                    c[hidx[i]] += 1
                end
            end
            return c ./ (4_000 * n)
        end

        # 6 prior sd covers essentially all the mass here, so the two agree
        @test maximum(abs, tally(Inf, 11) .- tally(6.0, 11)) < 0.01
    end

    @testset "conjugate updates" begin
        Random.seed!(7)
        n = 500
        y = randn(n)
        curref = randn(n)
        τobs = fill(3.0, n)
        prior = (0.5, 0.25)

        M = 20_000
        s = [G.sample_alpha(y, curref, τobs, prior, Random.default_rng()) for _ in 1:M]
        prec = prior[2] + sum(τobs)
        mu = (prior[2] * prior[1] + sum(τobs .* (y .- curref))) / prec
        @test mean(s) ≈ mu atol = 5 / sqrt(M * prec)
        @test std(s) ≈ 1 / sqrt(prec) rtol = 0.05

        # μ update against the dense conditional
        t = collect(1.0:20)
        ar = G.AR1(t; phi = 3.0, rho = 0.6)
        C = densecov(t, 3.0, 0.6)
        Σ = [2.0 0.5; 0.5 1.0]
        h = randn(20, 2)
        sc = 1 / sum(C \ ones(20))
        target = vec(sc * (ones(20)' * (C \ h)))
        acc = zeros(2); acc2 = zeros(2, 2)
        μ = zeros(2)
        for _ in 1:M
            G.sample_mu!(μ, h, ar, Σ, Random.default_rng())
            acc .+= μ
            acc2 .+= (μ - target) * (μ - target)'
        end
        @test acc ./ M ≈ target atol = 0.03
        @test acc2 ./ M ≈ sc * Σ rtol = 0.06
    end

    @testset "end-to-end recovery" begin
        Random.seed!(8)
        n = 400
        spacing = 60.0
        t = spacing .* (1:n)
        g = OffsetGrid(-12.0:1.0:12.0, -12.0:1.0:12.0)
        k = length(g)

        Sg = GEDIGEOLOC.expcor(g.coords, 8.0) + 1e-8I
        refvals = cholesky(Symmetric(Sg)).L * randn(k, n)

        ϕtrue, ρtrue = 4_000.0, 0.9
        Σtrue = 12.0 * [1.0 0.5; 0.5 1.0]
        μtrue = [3.0, -2.0]
        Ctrue = densecov(collect(t), ϕtrue, ρtrue)
        off = μtrue' .+ cholesky(Symmetric(Ctrue)).L * randn(n, 2) * cholesky(Σtrue).U

        idx = [nearestindex(g, off[i, 1], off[i, 2]) for i in 1:n]
        snapped = g.coords[idx, :]
        τtrue = 100.0
        yobs = [refvals[idx[i], i] for i in 1:n] .+ randn(n) ./ sqrt(τtrue)

        priors = (α = (0.0, 100.0), τ = gammashaperate(100.0, 20.0),
                  Σ = (20.0, diagm(fill(12.0, 2))),
                  ϕ = gammashaperate(4_000.0, 1_500.0), ρ = (2.5, 1.0))
        initial = (w = zeros(n, 2), Σ = diagm(fill(12.0, 2)), α = 0.0,
                   ϕ = 4_000.0, ρ = 0.9, μ = [0.0, 0.0], τ = 100.0)

        s = geolocate(yobs, t, g, refvals, priors, initial, 600;
                      burn = 300, progress = false, rng = Xoshiro(99))

        hm = posteriormean(s)
        @test cor(vec(hm), vec(snapped)) > 0.9
        @test sqrt(mean(abs2, hm .- snapped)) < 3.0
        @test mean(s.α) ≈ 0.0 atol = 0.5
        @test 0.0 < s.accept < 1.0
        @test size(s.hidx) == (n, 600)
        @test posteriormean(s) ≈ dropdims(mean(s.h, dims = 3), dims = 3)

        # Float32 reference must give the same answer to within noise
        s32 = geolocate(yobs, t, g, Float32.(refvals), priors, initial, 600;
                        burn = 300, progress = false, rng = Xoshiro(99))
        @test sqrt(mean(abs2, posteriormean(s32) .- snapped)) < 3.0

        # threading must not change the target
        s2 = geolocate(yobs, t, g, refvals, priors, initial, 600;
                       burn = 300, progress = false, nchunks = 4, rng = Xoshiro(7))
        @test sqrt(mean(abs2, posteriormean(s2) .- snapped)) < 3.0

        # windowing must not change the answer materially here
        sw = geolocate(yobs, t, g, refvals, priors, initial, 600;
                       burn = 300, progress = false, window = 6.0, rng = Xoshiro(99))
        @test sqrt(mean(abs2, posteriormean(sw) .- snapped)) < 3.0

        # mixture variant
        zt = rand(n) .< 0.05
        ymix = copy(yobs)
        ymix[zt] .+= randn(count(zt))
        pmix = (α = (0.0, 100.0), τ₀ = gammashaperate(100.0, 20.0),
                τ₁ = gammashaperate(1.0, 0.5), πz = betaparams(0.05, 0.05),
                Σ = (20.0, diagm(fill(12.0, 2))),
                ϕ = gammashaperate(4_000.0, 1_500.0), ρ = (2.5, 1.0))
        imix = merge(initial, (τ = [100.0, 1.0], πz = 0.05, z = zt))
        smix = geolocate(ymix, t, g, refvals, pmix, imix, 400;
                         mixture = true, burn = 200, progress = false, rng = Xoshiro(5))
        @test sqrt(mean(abs2, posteriormean(smix) .- snapped)) < 3.5
        @test mean(smix.z[zt, :]) > mean(smix.z[.!zt, :])

        # nugget-free variant
        snn = geolocate(yobs, t, g, refvals, priors, initial, 300;
                        nugget = false, burn = 200, progress = false, rng = Xoshiro(3))
        @test snn.w === nothing
        @test all(snn.ρ .== 1.0)
        @test sqrt(mean(abs2, posteriormean(snn) .- snapped)) < 3.5

        # prediction at held-out coordinates
        tp = collect(t[2:2:20]) .- spacing / 2
        op, gi = predoffsets(s, tp, t, g; every = 5)
        @test size(op) == (length(tp), 2, length(1:5:600))
        @test all(1 .≤ gi .≤ k)
        pm = dropdims(mean(op, dims = 3), dims = 3)
        neigh = (off[2:2:20, :] .+ off[1:2:19, :]) ./ 2
        @test cor(vec(pm), vec(neigh)) > 0.8
    end

    @testset "predoffsets bridge matches the dense conditional" begin
        Random.seed!(9)
        n, ϕ, ρ = 9, 2.5, 1.0
        t = collect(1.0:n)
        tp = [2.4, 5.5, 0.3, 12.0]          # interior, and outside both ends
        g = OffsetGrid(-5.0:1.0:5.0, -5.0:1.0:5.0)
        wtrue = randn(n, 2)
        Σ = [1.0 0.0; 0.0 1.0]

        M = 40_000
        samples = (w = repeat(wtrue, 1, 1, M), ρ = fill(ρ, M), ϕ = fill(ϕ, M),
                   μ = zeros(2, M), Σ = repeat(Σ, 1, 1, M), grid = g)
        op, _ = predoffsets(samples, tp, t, g)
        emp = dropdims(mean(op, dims = 3), dims = 3)

        Kxx = densecor(t, ϕ)
        Kpx = [exp(-abs(a - b) / ϕ) for a in tp, b in t]
        target = Kpx * (Kxx \ wtrue)
        @test emp ≈ target atol = 0.03
    end

    @testset "argument validation" begin
        g = OffsetGrid(-1.0:1.0:1.0, -1.0:1.0:1.0)
        k, n = length(g), 5
        R = randn(k, n)
        y = randn(n)
        t = collect(1.0:n)
        pri = (α = (0.0, 1.0), τ = (1.0, 1.0), Σ = (5.0, diagm(fill(1.0, 2))),
               ϕ = (1.0, 1.0), ρ = (2.0, 1.0))
        ini = (Σ = diagm(fill(1.0, 2)), α = 0.0, ϕ = 1.0, μ = [0.0, 0.0], τ = 1.0)

        @test_throws DimensionMismatch geolocate(y, t, g, randn(k, n + 1), pri, ini, 5; progress = false)
        @test_throws DimensionMismatch geolocate(y, t[1:3], g, R, pri, ini, 5; progress = false)
        @test_throws ArgumentError geolocate(y, t, g, R, (α = (0.0, 1.0),), ini, 5; progress = false)
        @test_throws ArgumentError geolocate(y, t, g, R, pri, (α = 0.0,), 5; progress = false)
        @test_throws ArgumentError G.AR1(reverse(t); phi = 1.0, rho = 0.5)
        @test_throws DomainError G.AR1(t; phi = 1.0, rho = 1.5)
    end
end
