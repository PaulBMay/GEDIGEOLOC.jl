"""
    OffsetGrid(coords)
    OffsetGrid(x, y)
    OffsetGrid(halfwidth; step = 1.0)

The `k × 2` set of candidate geolocation offsets.  Row `j` of `coords`
corresponds to row `j` of the `reference` matrix, i.e. `reference[j, i]` is the
reference (DEM) value for footprint `i` displaced by `coords[j, :]`.

When the offsets form a regular lattice — as they do whenever the reference
values are extracted by shifting a raster — the grid records the lattice
parameters.  That buys two things:

* `nearestindex` in `O(1)` arithmetic instead of a k-way search;
* the `window` option of [`geolocate`](@ref), which restricts each footprint's
  categorical draw to the lattice cells the prior actually supports.

`OffsetGrid(coords)` recovers the lattice from a plain matrix when it can, so
grids built by the usual double loop keep working unchanged.  The expected row
order is x-major / y-minor: `coords[j, :] == [x[a], y[b]]` with
`j = (a-1)*ny + b`.
"""
struct OffsetGrid{T<:AbstractFloat}
    coords::Matrix{T}
    x0::T
    dx::T
    nx::Int
    y0::T
    dy::T
    ny::Int
    lattice::Bool
end

Base.length(g::OffsetGrid) = size(g.coords, 1)
Base.size(g::OffsetGrid) = size(g.coords)
islattice(g::OffsetGrid) = g.lattice

function OffsetGrid(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    T = float(promote_type(eltype(x), eltype(y)))
    nx, ny = length(x), length(y)
    coords = Matrix{T}(undef, nx * ny, 2)
    @inbounds for a in 1:nx, b in 1:ny
        j = (a - 1) * ny + b
        coords[j, 1] = x[a]
        coords[j, 2] = y[b]
    end
    dx = nx > 1 ? T(x[2] - x[1]) : one(T)
    dy = ny > 1 ? T(y[2] - y[1]) : one(T)
    return OffsetGrid{T}(coords, T(x[1]), dx, nx, T(y[1]), dy, ny, true)
end

OffsetGrid(halfwidth::Real; step::Real = 1.0) =
    OffsetGrid(-halfwidth:step:halfwidth, -halfwidth:step:halfwidth)

OffsetGrid(g::OffsetGrid) = g

function OffsetGrid(coords::AbstractMatrix{<:Real})
    size(coords, 2) == 2 || throw(ArgumentError("grid must have two columns (x and y offsets)"))
    T = float(eltype(coords))
    C = convert(Matrix{T}, coords)
    lat = _detect_lattice(C)
    lat === nothing && return OffsetGrid{T}(C, zero(T), one(T), 0, zero(T), one(T), 0, false)
    x0, dx, nx, y0, dy, ny = lat
    return OffsetGrid{T}(C, x0, dx, nx, y0, dy, ny, true)
end

# Recognise the x-major/y-minor lattice produced by the usual double loop.
function _detect_lattice(C::Matrix{T}) where {T}
    k = size(C, 1)
    k ≥ 1 || return nothing
    ny = something(findfirst(i -> C[i, 1] != C[1, 1], 1:k), k + 1) - 1
    ny ≥ 1 || return nothing
    (nx, rem) = divrem(k, ny)
    rem == 0 || return nothing

    x0, y0 = C[1, 1], C[1, 2]
    dx = nx > 1 ? C[ny+1, 1] - x0 : one(T)
    dy = ny > 1 ? C[2, 2] - y0 : one(T)
    (isfinite(dx) && dx != 0 && isfinite(dy) && dy != 0) || return nothing

    tol = 1e-8 * max(abs(dx), abs(dy))
    @inbounds for a in 1:nx, b in 1:ny
        j = (a - 1) * ny + b
        abs(C[j, 1] - (x0 + (a - 1) * dx)) ≤ tol || return nothing
        abs(C[j, 2] - (y0 + (b - 1) * dy)) ≤ tol || return nothing
    end
    return (x0, dx, nx, y0, dy, ny)
end

"""
    nearestindex(g::OffsetGrid, px, py)

Row index of the grid point closest to `(px, py)`.  `O(1)` on a lattice,
`O(k)` otherwise.
"""
function nearestindex(g::OffsetGrid, px::Real, py::Real)
    if g.lattice
        a = clamp(round(Int, (px - g.x0) / g.dx) + 1, 1, g.nx)
        b = clamp(round(Int, (py - g.y0) / g.dy) + 1, 1, g.ny)
        return (a - 1) * g.ny + b
    end
    C = g.coords
    best = 1
    bestd = Inf
    @inbounds for j in axes(C, 1)
        d = (C[j, 1] - px)^2 + (C[j, 2] - py)^2
        if d < bestd
            best = j
            bestd = d
        end
    end
    return best
end

"""
    centerindex(g::OffsetGrid)

Row index of the grid point closest to the origin — the natural starting value
for the offsets.
"""
centerindex(g::OffsetGrid) = nearestindex(g, zero(eltype(g.coords)), zero(eltype(g.coords)))
