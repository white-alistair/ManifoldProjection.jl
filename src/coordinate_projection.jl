struct CoordinateProjection{R<:Function,J<:Function,T<:AbstractFloat,M<:Integer}
    residual_function::R
    residual_jacobian::J
    tolerance::T
    maxiters::M
end

function CoordinateProjection(
    residual_function::R,
    residual_jacobian::J,
    tolerance::T,
) where {R,J,T}
    CoordinateProjection{R,J,T,Int64}(residual_function, residual_jacobian, tolerance, 10)
end

function coordinate_projection(
    u0::Vector{T},
    residual_function::Function,  # Function of u only
    jac::AbstractMatrix{T},       # Jacobian evaluated at (u0, t0)
    tolerance::T,
    maxiters::Integer,
) where {T}
    N = size(residual_function(u0))[1]     # Number of invariants
    λ = zeros(T, N)                        # One Lagrange multiplier for each invariant
    u = u0                                 # u will be the projection of u0 onto the manifold

    for _ = 1:maxiters
        residuals = residual_function(u)
        if all(abs.(residuals) .< tolerance)
            return u
        end
        λ -= inv(jac * jac') * residuals
        u = vec(u0 + jac' * λ)
    end

    @warn "maxiters exceeded for manifold projection, returning unprojected value"
    return u0
end

function (cp::CoordinateProjection{R,J,T,M})(u0::Vector{T}, t0::T) where {R,J,T,M}
    (; residual_function, residual_jacobian, tolerance, maxiters) = cp
    res = u -> residual_function(u, t0)
    jac = residual_jacobian(u0, t0)
    return coordinate_projection(u0, res, jac, tolerance, maxiters)
end
