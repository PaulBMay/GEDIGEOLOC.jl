# Exponential correlation function : cov(loc1, loc2) = exp(-dist(loc1, loc2) / range)
# Gives exponential decay in correlation with distance between locations.

# Single location set
function expcor(loc::AbstractMatrix, range::Number)
    return exp.( -pairwise(Euclidean(), loc, dims = 1) ./ range )
end


# Cross locations
function expcor(loc1::AbstractMatrix, loc2::AbstractMatrix, range::Number)
    return exp.( -pairwise(Euclidean(), loc1, loc2, dims = 1) ./ range )
end