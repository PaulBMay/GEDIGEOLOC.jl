# The three near-identical samplers `geolocate`, `geolocate2` and `geolocate_gfe`
# are now one function with two keywords.  These shims keep existing analysis
# scripts running.

"""
    geolocate2(y, t, grid, reference, priors, initial, nsamps; ...)

Deprecated.  Use `geolocate(...; nugget = false)`.
"""
function geolocate2(y, t, grid, reference, priors, initial, nsamps;
                    propsd = 1e-2, fixϕ = false, fixall = false, kwargs...)
    Base.depwarn("geolocate2(...) is deprecated; use geolocate(...; nugget = false)",
                 :geolocate2)
    fixall && throw(ArgumentError(
        "`fixall` is no longer supported; use `sampleϕ = false` together with " *
        "fixed `initial` values, or hold parameters fixed in the priors"))
    return geolocate(y, t, grid, reference, priors, initial, nsamps;
                     nugget = false, sampleϕ = !fixϕ, propsd = propsd, kwargs...)
end

"""
    geolocate_gfe(y, t, grid, reference, priors, initial, nsamps; ...)

Deprecated.  Use `geolocate(...; mixture = true)`.
"""
function geolocate_gfe(y, t, grid, reference, priors, initial, nsamps;
                       numnb = nothing, propvar = 1e-4, kwargs...)
    Base.depwarn("geolocate_gfe(...) is deprecated; use geolocate(...; mixture = true)",
                 :geolocate_gfe)
    numnb === nothing || @warn "`numnb` is ignored: the ν = 1/2 prior is exactly AR(1), " *
                               "so the precision is tridiagonal by construction." maxlog = 1
    return geolocate(y, t, grid, reference, priors, initial, nsamps;
                     mixture = true, propsd = sqrt(_scalarvar(propvar)), kwargs...)
end

_scalarvar(v::Number) = v
_scalarvar(v::AbstractMatrix) = v[1, 1]
_scalarvar(v::AbstractVector) = v[1]
