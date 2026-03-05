function predoffsets(samples, locp, loc, grid; every = 1)

    np = size(locp, 1)
    nsamps_og = length(samples.ϕ)
    samp_idx = 1:every:nsamps_og
    nsamps = length(samp_idx)

    nb = predneighbors(vec(locp), vec(loc))

    offsets = zeros(np, 2, nsamps)

    tree = KDTree(grid')
    grididx = zeros(Int, np, nsamps)

    @views for m in ProgressBar(1:nsamps)

        mm = samp_idx[m]

        ρ = samples.ρ[mm]
        ϕ = samples.ϕ[mm]
        μ = samples.μ[:,mm]
        Σ = samples.Σ[:,:,mm]
        Σc = cholesky(Σ)
        w = samples.w[:,:,mm]

        wu = w / Σc.U

        B, F = getnngppred(nb, locp, loc, ϕ)

        hu_samp = B*wu + # Conditional mean 
            sqrt.(ρ * F) .* randn(np, 2) + # Conditional error of w
            sqrt(1 - ρ) * randn(np, 2) # (Un)conditional error of ϵ

        h_samp = μ' .+ (hu_samp * Σc.U)

        offsets[:,:,m] .= h_samp

        for i in 1:np
            grididx[i,m] = knn(tree, h_samp[i,:], 1, false)[1][1]
        end

    end

    return offsets, grididx


end