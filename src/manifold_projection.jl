struct HairerProjection{C<:Function, J<:Function, T<:AbstractFloat, M<:Integer}
    constraints::C
    jacobian::J
    tolerance::T
    maxiters::M
end

function HairerProjection(constraints::C, jacobian::J, tolerance::T) where {C,J,T}
    HairerProjection{C,J,T,Int64}(constraints, jacobian, tolerance, 10)
end

function (manifold_projection::HairerProjection{C,J,T,M})(u0::Vector{T}) where {C,J,T,M}
    (; constraints, jacobian, tolerance, maxiters) = manifold_projection

    N = size(constraints(u0))[1]  # Number of constraints
    λ = zeros(T, N)               # One Lagrange multiplier for each constraint
    u = u0                        # u will be the projection of u0 on to the manifold

    for _ in 1:maxiters
        residuals = constraints(u)
        if all(abs.(residuals) .< tolerance)
            return u
        end
        λ -= inv(jacobian(u0) * jacobian(u0)') * residuals
        u = vec(u0 + jacobian(u0)' * λ)
    end

    @warn "maxiters exceeded for manifold projection, returning unprojected value"
    return u0
end
