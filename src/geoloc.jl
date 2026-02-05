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
        ρpriordist = Normal(priors.ρ[1], priors.ρ[2])
        lprior = logpdf(ϕpriordist, ϕ) + logpdf(ρpriordist, theta[1])
        lpriorprop = logpdf(ϕpriordist, ϕprop) + logpdf(ρpriordist, thetaprop[1])

        acceptprob = exp(llprop + lpriorprop + thetaprop[2] - ll - lprior - theta[2])

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


function geolocate2(y, time, grid, reference, priors, initial, nsamps; propsd = 1e-2, fixϕ = false, fixall = false)


    #######################
    # Helper funcs
    #######################

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
    
    function llϕ(res, Q, Σ)

        n,p = size(res)
    
        Σc = cholesky(Σ)
    
        resS = ( Q * res ) / Σc
    
        sse = vec(res)'*vec(resS)
    
        ldet = n*logdet(Σc) - p*logdet(Q)
    
        ll = -0.5*(ldet + sse)
    
        return ll
    
    end


    ######################
    # Gibbs components
    ######################

    function sample_h!(h, curref, loglike, gridu, reference, grid, y, α, τ, μ, Σ, Q)

        n = size(h,1)
        
        loglike .= @. -0.5*τ*(reference + α - y')^2

        Σchol = cholesky(Σ)

        gridu .= grid / Σchol.U

        # Sample first footprint

        w_next = h[2,:] - μ
        w_prev = similar(w_next) # No previous, yet

        meanu_i = Σchol.L \ (μ - ( Q.ev[1]*w_next ) ./ Q.dv[1] )

        logprior = -0.5*vec(sum( (gridu .-  meanu_i').^2, dims = 2)) .* Q.dv[1]
        logprob = view(loglike, :, 1) + logprior
        prob = softmax(logprob)
        hind = rand(Categorical(prob))
        h[1,:] = grid[hind,:]
        curref[1] = reference[hind, 1]

        # Sample footprints 2:n

        @inline @fastmath for i in 2:(n-1)

            w_next .= h[i+1,:] - μ
            w_prev .= h[i-1,:] - μ
            meanu_i .= Σchol.L \ (μ - ( Q.ev[i]*w_next + Q.ev[i-1]*w_prev) ./ Q.dv[i] )
            logprior .= -0.5*vec(sum( (gridu .-  meanu_i').^2, dims = 2)) .* Q.dv[i]
            logprob .= view(loglike, :, i) + logprior
            softmax!(prob, logprob)
            hind = rand(Categorical(prob))
            h[i,:] .= grid[hind,:]
            curref[i] = reference[hind, i]

        end

        # Sample last footprint

        w_prev .= h[n-1,:] - μ
        meanu_i .= Σchol.L \ (μ - (Q.ev[n-1]*w_prev) ./ Q.dv[n] )
        logprior .= -0.5*vec(sum( (gridu .-  meanu_i').^2, dims = 2)) .* Q.dv[n]
        logprob .= view(loglike, :, n) + logprior
        softmax!(prob, logprob)
        hind = rand(Categorical(prob))
        h[n,:] .= grid[hind,:]
        curref[n] = reference[hind, n]

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

    function sample_Σ(h, μ, Q, priors)
        res = h .- μ'
        resS = Q * res
        SSE = res' * resS
        freedom = priors.Σ[1] + size(h,1)
        scale = Distributions.PDMats.PDMat(Symmetric(priors.Σ[1]*priors.Σ[2] + SSE)) 
        return rand(InverseWishart(freedom, scale))
    end

    function sample_μ(h, Q, Σ)
        n = size(h,1)
        oneSolve = Q * ones(n)
        scale = 1/sum(oneSolve)
        μmu = vec(scale*(oneSolve'*h))
        return rand(MvNormal(μmu, Symmetric(scale*Σ)))
    end

    function sample_ϕ!(ρ, ρprop, Q, Qprop, lags, ϕ, h, μ, Σ, propsd, priors)

        res = h .- μ'

        logϕ = log(ϕ)
        logϕprop = logϕ + propsd*randn()
        ϕprop = exp(logϕprop)

        ρprop .= @. exp(-lags / ϕprop)
        formQar1!(Qprop, ρprop) 

        ll = llϕ(res, Q, Σ)
        llprop = llϕ(res, Qprop, Σ)

        ϕpriordist = Gamma(priors.ϕ[1], 1/priors.ϕ[2])
        lprior = logpdf(ϕpriordist, ϕ)
        lpriorprop = logpdf(ϕpriordist, ϕprop)

        acceptprob = exp(llprop + lpriorprop + logϕprop - ll - lprior - logϕ)

        if rand() < acceptprob
            ϕ = ϕprop
            Q.dv .= Qprop.dv
            Q.ev .= Qprop.ev
            ρ .= ρprop
        end

        return ϕ

    end
    

    ##########################



    n = length(y)

    # Open up sample memory and get initial values

    hsamps = zeros(n, 2, nsamps)
    h = initial.h
    hsamps[:,:,1] = h

    Σsamps = zeros(2, 2, nsamps)
    Σ = initial.Σ
    Σsamps[:,:,1] = Σ

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

    lags = time[2:n] - time[1:(n-1)]
    ρ = @. exp(-lags / ϕ)
    ρprop = similar(ρ)
    Q = formQar1(ρ)
    Qprop = similar(Q)

    loglike = similar(reference)
    gridu = similar(grid)

    ##########################################
    # Begin sampling
    ##########################################

    iter = ProgressBar(1:nsamps)

    for m in iter

        sample_h!(h, curref, loglike, gridu, reference, grid, y, α, τ, μ, Σ, Q)

        if !fixall
        α = sample_α(y, curref, τ, priors)

        τ = sample_τ(y, α, curref, priors)

        Σ = sample_Σ(h, μ, Q, priors)

        μ = sample_μ(h, Q, Σ)

        if !fixϕ ϕ = sample_ϕ!(ρ, ρprop, Q, Qprop, lags, ϕ, h, μ, Σ, propsd, priors) end

        end

        ######

        hsamps[:,:,m] .= h
        αsamps[m] = α
        τsamps[m] = τ
        Σsamps[:,:,m] .= Σ
        μsamps[:,m] .= μ
        ϕsamps[m] = ϕ


    end

    return (h = hsamps, α = αsamps, τ = τsamps, Σ = Σsamps, μ = μsamps, ϕ = ϕsamps)

end

function geolocate_gfe(y, loc, grid, reference, priors, initial, nsamps; numnb = 1, propvar = 1e-4)


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

    function sample_h!(h, loglike, meanu, gridu, grid, y, curref, α, reference, τ, z, μ, w, ρ, Σ)

        n = size(h,1)
        
        logprior = zeros(size(reference,1))
        logprob = zeros(size(reference, 1))
        prob = zeros(size(reference, 1))

        loglike .= @. -0.5 * τ[z+1]' * (reference + α - y')^2

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

    function sample_α(y, curref, τ, z, priors)
        τn = τ[z .+ 1]
        postprec = priors.α[2] + sum(τn)
        resid_norm = τn .* (y - curref)
        postmu = (sum(resid_norm) + priors.α[2]*priors.α[1])/postprec
        return postmu + sqrt(1/postprec)*randn()
    end

    function sample_τ(y, α, curref, z, priors)

        resid² = @. (y - α - curref)^2

        shape₀ = priors.τ₀[1] + 0.5*sum(.!z)
        rate₀ = priors.τ₀[2] + 0.5*sum(resid²[.!z])
        τ₀ = rand(Gamma(shape₀, 1 / rate₀))

        shape₁ = priors.τ₁[1] + 0.5*sum(z)
        rate₁ = priors.τ₁[2] + 0.5*sum(resid²[z])
        τ₁ = rand(Gamma(shape₁, 1 / rate₁))

        return [τ₀, τ₁]
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
        ρpriordist = Normal(priors.ρ[1], priors.ρ[2])
        lprior = logpdf(ϕpriordist, ϕ) + logpdf(ρpriordist, theta[1])
        lpriorprop = logpdf(ϕpriordist, ϕprop) + logpdf(ρpriordist, thetaprop[1])

        acceptprob = exp(llprop + lpriorprop + thetaprop[2] - ll - lprior - theta[2])

        if rand() < acceptprob
            ρ, ϕ = ρprop, ϕprop
            Qpostchol = copy(Qpostpropchol)
            nngp.B.nzval .= nngpprop.B.nzval
            nngp.F .= nngpprop.F
        end

        return ρ, ϕ, Qpostchol

    end

    function sample_z(y, α, curref, τ, πz)

        resid² = @. (y - α - curref)^2
        ll0 = @. -0.5 * ( τ[1] * resid² - log(τ[1]) ) + log(1 - πz)
        ll1 = @. -0.5 * ( τ[2] * resid² - log(τ[2]) ) + log(πz)

        maxll = max.(ll0, ll1)

        ll0 .-= maxll
        ll1 .-= maxll

        πz_post = @. exp(ll1) / (exp(ll1) + exp(ll0))
        
        return BitVector([rand(Bernoulli(p)) for p in πz_post])

    end

    function sample_πz(z, priors)

        sumz = sum(z)
        απz_post = priors.πz[1] + sumz
        βπz_post = priors.πz[2] + length(z) - sumz

        return rand(Beta(απz_post, βπz_post))

    end
    
    

    ##########################



    n = length(y)
    k = size(grid,1)

    # Open up sample memory and get initial values

    hsamps = zeros(n, 2, nsamps)
    h = zeros(n,2)

    wsamps = zeros(n, 2, nsamps)
    w = copy(initial.w)
    wsamps[:,:,1] .= initial.w

    Σsamps = zeros(2, 2, nsamps)
    Σ = copy(initial.Σ)
    Σsamps[:,:,1] .= Σ

    ρsamps = zeros(nsamps)
    ρ = initial.ρ
    ρsamps[1] = ρ

    ϕsamps = zeros(nsamps)
    ϕ = initial.ϕ
    ϕsamps[1] = ϕ

    μsamps = zeros(2, nsamps)
    μ = copy(initial.μ)
    μsamps[:,1] .= μ

    αsamps = zeros(nsamps)
    α = initial.α
    αsamps[1] = α

    τsamps = zeros(2, nsamps)
    τ = copy(initial.τ)
    τsamps[:,1] .= τ

    zsamps = BitMatrix(undef, n, nsamps)
    z = BitVector(initial.z)
    zsamps[:,1] .= z

    πzsamps = zeros(nsamps)
    πz = initial.πz
    πzsamps[1] = πz

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

        sample_h!(h, loglike, meanu, gridu, grid, y, curref, α, reference, τ, z, μ, w, ρ, Σ)

        α = sample_α(y, curref, τ, z, priors)

        τ .= sample_τ(y, α, curref, z, priors)

        z .= sample_z(y, α, curref, τ, πz)

        πz = sample_πz(z, priors)

        Σ .= sample_Σ(h, μ, ρ, Qpostchol, priors)

        μ .= sample_μ(h, Qpostchol, ρ, Σ)

        sample_w!(w, h, μ, ρ, Σ, Qpostchol)

        ρ, ϕ, Qpostchol = sample_ρϕ!(nngp, nngpprop, Qpostchol, Qpostpropchol, numnb, loc, ρ, ϕ, h, μ, Σ, propvarc, priors)

        ######

        hsamps[:,:,m] .= h
        αsamps[m] = α
        τsamps[:,m] .= τ
        zsamps[:,m] .= z
        πzsamps[m] = πz
        Σsamps[:,:,m] .= Σ
        μsamps[:,m] .= μ
        wsamps[:,:,m] .= w
        ρsamps[m] = ρ
        ϕsamps[m] = ϕ


    end

    return (h = hsamps, α = αsamps, τ = τsamps, z = zsamps, Σ = Σsamps, μ = μsamps, w = wsamps, ρ = ρsamps, ϕ = ϕsamps)

end