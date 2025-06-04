function getnngp(loc::AbstractArray, ϕ::Real, m::Integer)

    local n = size(loc,1)

    local Bnnz = sum(1:(m+1)) + (n - m - 1)*(m+1)
    local Bvals = zeros(Bnnz)
    Bvals[1] = 1.0

    local Brows = zeros(Int64, Bnnz)
    local Bcols = zeros(Int64, Bnnz)
    Brows[1] = 1.0 
    Bcols[1] = 1.0

    local Fvals = zeros(n)
    Fvals[1] = 1.0

    curInd = 1

    @views for i in 2:(m+1)

        indi = 1:(i-1)

        mi = i - 1

        rho = expcor(loc[indi,:], ϕ)
        k = expcor(loc[indi,:], loc[[i],:], ϕ)

        cholesky!(rho) 

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = 1.0 - sum(k.^2)

        ldiv!(UpperTriangular(rho), k)

        Bvals[(curInd+1):(curInd + mi + 1)] = [1; -k]
        Brows[(curInd+1):(curInd + mi + 1)] .= i
        Bcols[(curInd+1):(curInd + mi + 1)] = [i; indi]

        curInd += mi + 1

    end

    # Allocate arrays for computation within loop
    local rho = zeros(m, m)
    local k = zeros(m,1)


    @views for i in (m+2):n
        
        indi = (i-m):(i-1)

        rho .= expcor(loc[indi,:], ϕ)
        k .= expcor(loc[indi,:], loc[[i],:], ϕ)

        cholesky!(rho) 

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = 1 - sum(k.^2)

        ldiv!(UpperTriangular(rho), k)

        Bvals[(curInd+1):(curInd + m + 1)] = [1; -k]
        Brows[(curInd+1):(curInd + m + 1)] .= i
        Bcols[(curInd+1):(curInd + m + 1)] = [i; indi]


        curInd += m + 1

    end

    B = sparse(Brows, Bcols, Bvals)

    Border = invperm(sortperm( @.(Bcols + ( Brows ./ (n+1) ))))

    return NNGP(B, Fvals, Border)

end

function getnngp!(nngp::NNGP, loc::AbstractArray, ϕ::Real, m::Integer)

    n = size(loc,1)

    curInd = 1

    @views for i in 2:(m+1)

        indi = 1:(i-1)

        mi = i - 1

        rho = expcor(loc[indi,:], ϕ)
        k = expcor(loc[indi,:], loc[[i],:], ϕ)

        cholesky!(rho) 

        ldiv!(UpperTriangular(rho)', k)

        nngp.F[i] = 1.0 - sum(k.^2)

        ldiv!(UpperTriangular(rho), k)

        valIndex = nngp.Border[(curInd+1):(curInd + mi + 1)]

        nngp.B.nzval[valIndex] = [1; -k]

        curInd += mi + 1

    end

    # Allocate arrays for computation within loop
    local rho = zeros(m, m)
    local k = zeros(m,1)


    @views for i in (m+2):n
        
        indi = (i-m):(i-1)

        rho .= expcor(loc[indi,:], ϕ)
        k .= expcor(loc[indi,:], loc[[i],:], ϕ)

        cholesky!(rho) 

        ldiv!(UpperTriangular(rho)', k)

        nngp.F[i] = 1.0 - sum(k.^2)

        ldiv!(UpperTriangular(rho), k)

        valIndex = nngp.Border[(curInd+1):(curInd + m + 1)]

        nngp.B.nzval[valIndex] = [1; -k]

        curInd += m + 1

    end

    return nothing

end

function formprec(nngp::NNGP)
    return nngp.B' * (Diagonal(1 ./ nngp.F) * nngp.B)
end

function simnngp(nngp::NNGP)

    n = length(nngp.F)

    return LowerTriangular(nngp.B) \ (sqrt.(nngp.F) .* randn(n))

end

function Base.copy(nngp::NNGP)
    return NNGP(copy(nngp.B), copy(nngp.F), copy(nngp.Border))
end