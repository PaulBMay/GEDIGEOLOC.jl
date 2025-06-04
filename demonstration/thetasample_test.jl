using GEDIGEOLOC

using LinearAlgebra, SparseArrays
using Random, Distributions
using Plots

using NearestNeighbors

using ProgressBars

######################

    function logit(x)
        return log(x) - log(1-x)
    end

    function softmax(x)
        return 1/(1+exp(-x))
    end

    function llρϕ(res, Qpostchol, nngp, Σ, ρ)

        n,p = size(res)
    
        Σc = cholesky(Σ)
    
        resS = ( (1/(1-ρ))*res - (1/(1-ρ)^2) * (Qpostchol \ res) ) / Σc
    
        sse = vec(res)'*vec(resS)
    
        ldet = n*logdet(Σc) + p*( logdet(Qpostchol) + sum(log.(nngp.F)) + n*log(ρ) + n*log(1-ρ) )
    
        ll = -0.5*(ldet + sse)
    
        return ll
    
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


##############################

n = 1_000

loc = reshape(collect(60*(1:n)), :, 1)

ϕtrue = 1000
ρtrue = 0.95
Σtrue = [2 1; 1 2]
μtrue = [1, -5]

numnb = 1

nngptrue = getnngp(loc, ϕtrue, numnb)
Σtruechol = cholesky(Σtrue)

Random.seed!(96)

w = sqrt(ρtrue) * (
    LowerTriangular(nngptrue.B) \ (sqrt.(nngptrue.F) .* randn(n,2))
) * Σtruechol.U


plot(loc, w[:,1])
plot!(loc, w[:,2])

h = μtrue' .+ w + sqrt(1-ρtrue)*randn(n,2)*Σtruechol.U

#################################
##################################

nngp = copy(nngptrue)
nngpprop = copy(nngp)

ρ, ϕ, μ, Σ = ρtrue, ϕtrue, copy(μtrue), copy(Σtrue)

Qpostchol = cholesky(
    (1/ρ)*formprec(nngp) + (1/(1-ρ))*I(n)
)
Qpostcholprop = copy(Qpostchol)

propvarc = cholesky(1e-1*Matrix(I,2,2))

priors = (ϕ = gammashaperate(ϕtrue, 500), α = (0, 100))

#########################3
nsamps = 1000

ϕsamps, ρsamps = zeros(nsamps), zeros(nsamps)


for m in ProgressBar(1:nsamps)

    ϕ
    sum(log.(nngp.F))
    logdet(Qpostchol)

    ρ, ϕ, Qpostchol = sample_ρϕ!(nngp, nngpprop, Qpostchol, Qpostcholprop, numnb, loc, ρ, ϕ, h, μ, Σ, propvarc, priors)

    ρsamps[m], ϕsamps[m] = ρ, ϕ

end

scatter(ρsamps)
scatter(ϕsamps)