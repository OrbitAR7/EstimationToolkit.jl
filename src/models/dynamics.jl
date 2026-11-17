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

A model whose process noise depends on the state implements
`process_noise_cov(model, x, dt)` instead (or as well); see [`process_noise_cov`](@ref).
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
    process_noise_cov(model::AbstractDynamicsModel, x, dt)

Return the process noise covariance matrix Q for time step `dt`.

A model defines one of the two forms:

- `process_noise_cov(model, dt)` when Q does not depend on the state. This is the
  usual case, and the three-argument form then falls back to it.
- `process_noise_cov(model, x, dt)` when Q depends on the state `x` at the start
  of the step (for example density uncertainty that scales with descent rate, or
  drag noise that scales with dynamic pressure).

The Kalman-type filters (`KalmanFilter`, `InformationFilter`,
`SquareRootInformationFilter`, `SquareRootCovarianceFilter`, `FadingMemoryFilter`,
the EKF family, and the UKF, cubature, quadrature and divided-difference filters)
call the three-argument form in `predict!`, with `x` the state estimate before
propagation, so Q is evaluated at the same point as the Jacobian. `x` is passed
as the filter stores it (an `SVector` for most filters).

The particle, ensemble, grid and Gaussian-mixture filters, `sample_process_noise`
and `simulate_trajectory` still call `process_noise_cov(model, dt)`. A model that
is used with those as well should define both forms.

# Example
```julia
struct Descent <: AbstractDynamicsModel{2, 0, 2} end

# x = [altitude, descent rate]; the noise on the rate grows with the rate
EstimationToolkit.process_noise_cov(::Descent, x, dt) =
    SMatrix{2,2}(1e-6 * dt, 0.0, 0.0, (0.05 * x[2])^2 * dt)
```
"""
function process_noise_cov end

# State-independent models: the state-aware form ignores the state.
process_noise_cov(model::AbstractDynamicsModel, x, dt) = process_noise_cov(model, dt)
