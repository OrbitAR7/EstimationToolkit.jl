# Automatic differentiation utilities for Jacobian computation

"""
    jacobian_ad(f, x)

Compute Jacobian of f at x using automatic differentiation.

This is a fallback for models that don't provide analytical Jacobians.
"""
function jacobian_ad(f, x::SVector{N, T}) where {N, T}
    J = ForwardDiff.jacobian(f, x)
    return SMatrix{size(J, 1), size(J, 2), T}(J)
end

function jacobian_ad(f, x::AbstractVector{T}) where {T}
    return ForwardDiff.jacobian(f, x)
end

# Default implementations using AD for dynamics models
function jacobian_x(model::AbstractDynamicsModel, x::SVector, u, dt)
    f_wrap = x_var -> propagate(model, x_var, u, dt)
    return jacobian_ad(f_wrap, x)
end

# Default implementations using AD for measurement models
function jacobian_x(model::AbstractMeasurementModel, x::SVector)
    h_wrap = x_var -> measure(model, x_var)
    return jacobian_ad(h_wrap, x)
end
