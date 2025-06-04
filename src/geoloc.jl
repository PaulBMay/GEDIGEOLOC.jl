function geolocate(y, loc, grid, reference, priors, initial, nsamps; numnb = 1, propvar = 1e-4)


    #######################
    # Helper funcs
    #######################


    function llρϕ(res, Qpostchol, nngp, Σ, ρ)

        n,p = size(res)
    
        Σc = cholesky(Σ)
    
        resS = ( (1/(1-ρ))*res - (1/(1-ρ)^2) * (Qpostchol \ res) ) / Σc
    
        sse = vec(res)'*vec(resS)
    
        ldet = n*logdet(Σc) + p*( logdet(Qpostchol) + sum(log.(nngp.F)) + n*log(ρ) + n*log(1-ρ) )
    
        ll = -0.5*(ldet + sse)
    
        return ll
    
    end

    function logit(x::Number)
        return log(x) - log(1-x)
    end

    function softmax(x::Number)
        return 1/(1+exp(-x))
    end

    ######################
    # Gibbs components
    ######################

    function sample_h!(h, loglike, meanu, gridu, grid, y, curref, α, reference, τ, μ, w, ρ, Σ)

        n = size(h,1)
        
        logprior = zeros(size(reference,1))
        logprob = zeros(size(reference, 1))
        prob = zeros(size(reference, 1))

        loglike .= @. -0.5*τ*(reference + α - y')^2

        covchol = cholesky((1-ρ)*Σ)

        gridu .= grid / covchol.U
        meanu .= (w .+ μ') / covchol.U


        @inline @fastmath for i in 1:n

            meanui = view(meanu, i, :)
            logprior .= -0.5*vec(sum( (gridu .-  meanui').^2, dims = 2))
            loglikei = view(loglike, :, i)
            logprob .= loglikei + logprior
            softmax!(prob, logprob)
            hind = rand(Categorical(prob))
            h[i,:] .= grid[hind,:]
            curref[i] = reference[hind, i]

        end

        return nothing

    end

    function sample_α(y, curref, τ, priors)
        postprec = priors.α[2] + length(y)*τ
        postmu = (τ*sum(y - curref) + priors.α[2]*priors.α[1])/postprec
        return postmu + sqrt(1/postprec)*randn()
    end

    function sample_τ(y, α, curref, priors)
        shape = priors.τ[1] + length(y)/2
        rate = priors.τ[2] + sum((y .- α .- curref).^2)/2
        return rand(Gamma(shape, 1/rate))
    end

    function sample_Σ(h, μ, ρ, Qpostchol, priors)
        res = h .- μ'
        resS = (1/(1-ρ))*res - (1/(1-ρ))^2 * (Qpostchol \ res)
        SSE = res' * resS
        freedom = priors.Σ[1] + size(h,1)
        scale = Distributions.PDMats.PDMat(Symmetric(priors.Σ[1]*priors.Σ[2] + SSE)) 
        return rand(InverseWishart(freedom, scale))
    end

    function sample_μ(h, Qpostchol, ρ, Σ)
        n = size(h,1)
        oneSolve = (1/(1-ρ))*ones(n) - (1/(1-ρ))^2 * (Qpostchol \ ones(n))
        scale = 1/sum(oneSolve)
        μmu = vec(scale*(oneSolve'*h))
        return rand(MvNormal(μmu, Symmetric(scale*Σ)))
    end

    function sample_w!(w, h, μ, ρ, Σ, Qpostchol)   
        ΣU = cholesky(Σ).U
        resu = ( (h .- μ') ./ (1- ρ) ) / ΣU
        wmu = Qpostchol \ resu
        werror = view(Qpostchol.U \ randn(size(w)), Qpostchol.p, :)
        w .= (wmu + werror) * ΣU
    end

    function sample_ρϕ!(nngp, nngpprop, Qpostchol, Qpostpropchol, numnb, loc, ρ, ϕ, h, μ, Σ, propvarc, priors)

        theta = [logit(ρ), log(ϕ)]

        thetaprop = theta + propvarc.L*randn(2)

        ρprop = softmax(thetaprop[1])
        ϕprop = exp(thetaprop[2])

        getnngp!(nngpprop, loc, ϕprop, numnb)

        Qpostprop = (1/ρprop)*formprec(nngpprop) + (1/(1-ρprop))*I(size(loc,1))
        cholesky!(Qpostpropchol, Qpostprop)

        res = h .- μ'

        ll = llρϕ(res, Qpostchol, nngp, Σ, ρ)
        llprop = llρϕ(res, Qpostpropchol, nngpprop, Σ, ρprop)

        ϕpriordist = Gamma(priors.ϕ[1], 1/priors.ϕ[2])
        lprior = logpdf(ϕpriordist, ϕ)
        lpriorprop = logpdf(ϕpriordist, ϕprop)

        acceptprob = exp(llprop + lpriorprop + sum(thetaprop) - ll - lprior - sum(theta))

        if rand() < acceptprob
            ρ, ϕ = ρprop, ϕprop
            Qpostchol = copy(Qpostpropchol)
            nngp.B.nzval .= nngpprop.B.nzval
            nngp.F .= nngpprop.F
        end

        return ρ, ϕ, Qpostchol

    end
    

    ##########################



    n = length(y)
    k = size(grid,1)

    # Open up sample memory and get initial values

    hsamps = zeros(n, 2, nsamps)
    h = zeros(n,2)

    wsamps = zeros(n, 2, nsamps)
    w = initial.w
    wsamps[:,:,1] = initial.w

    Σsamps = zeros(2, 2, nsamps)
    Σ = initial.Σ
    Σsamps[:,:,1] = Σ

    ρsamps = zeros(nsamps)
    ρ = initial.ρ
    ρsamps[1] = ρ

    ϕsamps = zeros(nsamps)
    ϕ = initial.ϕ
    ϕsamps[1] = ϕ

    μsamps = zeros(2, nsamps)
    μ = initial.μ
    μsamps[:,1] = μ

    αsamps = zeros(nsamps)
    α = initial.α
    αsamps[1] = α

    τsamps = zeros(nsamps)
    τ = initial.τ
    τsamps[1] = τ

    gridcenterind = argmin(sum(grid.^2, dims = 2)[:,1])
    curref = reference[CartesianIndex.(fill(gridcenterind, n), 1:n)]

    # Initialize important mats and decomps

    nngp = getnngp(loc, ϕ, numnb)
    nngpprop = copy(nngp)
    Qpost = (1/ρ)*formprec(nngp) + (1/(1-ρ))*I(n)
    Qpostchol = cholesky(Qpost)
    Qpostpropchol = copy(Qpostchol)

    if size(propvar,1) == 1
        propvar = propvar*I(2)
    end

    propvarc = cholesky(Symmetric(propvar))

    gridu = copy(grid)
    meanu = copy(w)
    loglike = copy(reference)

    ##########################################
    # Begin sampling
    ##########################################

    iter = ProgressBar(1:nsamps)

    for m in iter

        sample_h!(h, loglike, meanu, gridu, grid, y, curref, α, reference, τ, μ, w, ρ, Σ)

        α = sample_α(y, curref, τ, priors)

        τ = sample_τ(y, α, curref, priors)

        Σ = sample_Σ(h, μ, ρ, Qpostchol, priors)

        μ = sample_μ(h, Qpostchol, ρ, Σ)

        sample_w!(w, h, μ, ρ, Σ, Qpostchol)

        ρ, ϕ, Qpostchol = sample_ρϕ!(nngp, nngpprop, Qpostchol, Qpostpropchol, numnb, loc, ρ, ϕ, h, μ, Σ, propvarc, priors)

        ######

        hsamps[:,:,m] .= h
        αsamps[m] = α
        τsamps[m] = τ
        Σsamps[:,:,m] .= Σ
        μsamps[:,m] .= μ
        wsamps[:,:,m] .= w
        ρsamps[m] = ρ
        ϕsamps[m] = ϕ


    end

    return (h = hsamps, α = αsamps, τ = τsamps, Σ = Σsamps, μ = μsamps, w = wsamps, ρ = ρsamps, ϕ = ϕsamps)

end