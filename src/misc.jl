function gammashaperate(mu, std)

    shape = @. (mu/std)^2
    rate = @. mu / (std ^2)

    return shape, rate

end