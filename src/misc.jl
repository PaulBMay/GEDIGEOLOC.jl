function gammashaperate(mu, std)

    shape = @. (mu/std)^2
    rate = @. mu / (std ^2)

    return shape, rate

end

function betaparams(mu, std)
    var = std^2
    temp = mu*(1 - mu)/var - 1
    α = mu * temp
    β = (1 - mu) * temp
    return α, β
end

function logitmeansd(a, b, p)

    la = logit(a)
    lb = logit(b)

    μ = (la + lb)/2

    f(σ) = cdf(Normal(μ, σ), lb) - cdf(Normal(μ, σ), la) - p

    σ_hat = find_zero(f, (1e-3, 10.0), Bisection())
    
    return [μ, σ_hat]

end