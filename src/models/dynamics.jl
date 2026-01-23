"""
    AbstractDynamicsModel{NX, NU, NV}

Abstract type for dynamics models.

Type parameters:
- `NX`: State dimension
- `NU`: Control input dimension (use 0 if no control)
- `NV`: Process noise dimension

All dynamics models should implement:
- `propagate(model, x, u, dt)` - Propagate state forward
- `jacobian_x(model, x, u, dt)` - Jacobian w.r.t. state
- `jacobian_v(model, x, u, dt)` - Jacobian w.r.t. process noise
- `process_noise_cov(model, dt)` - Process noise covariance Q
"""
abstract type AbstractDynamicsModel{NX, NU, NV} end

state_dim(::AbstractDynamicsModel{NX, NU, NV}) where {NX, NU, NV} = NX
control_dim(::AbstractDynamicsModel{NX, NU, NV}) where {NX, NU, NV} = NU
process_noise_dim(::AbstractDynamicsModel{NX, NU, NV}) where {NX, NU, NV} = NV

"""
    propagate(model::AbstractDynamicsModel, x, u, dt)

Propagate state x forward by time dt with control input u.

Returns: x_{k+1} = f(x_k, u_k, dt)
"""
function propagate end

"""
    jacobian_x(model::AbstractDynamicsModel, x, u, dt)

Compute the Jacobian of the dynamics with respect to the state.

Returns: F = ∂f/∂x evaluated at (x, u, dt)
"""
function jacobian_x end

"""
    jacobian_v(model::AbstractDynamicsModel, x, u, dt)

Compute the Jacobian of the dynamics with respect to process noise.

Returns: G = ∂f/∂v evaluated at (x, u, dt)

Default implementation assumes additive noise: G = I
"""
function jacobian_v(model::AbstractDynamicsModel{NX, NU, NV}, x, u, dt) where {NX, NU, NV}
    return SMatrix{NX, NV, Float64}(I(NX))
end

"""
    process_noise_cov(model::AbstractDynamicsModel, dt)

Return the process noise covariance matrix Q for time step dt.
"""
function process_noise_cov end
