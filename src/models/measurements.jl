"""
    AbstractMeasurementModel{NX, NY, NW}

Abstract type for measurement models.

Type parameters:
- `NX`: State dimension
- `NY`: Measurement dimension  
- `NW`: Measurement noise dimension

All measurement models should implement:
- `measure(model, x)` - Compute expected measurement
- `jacobian_x(model, x)` - Jacobian w.r.t. state
- `measurement_noise_cov(model)` - Measurement noise covariance R
"""
abstract type AbstractMeasurementModel{NX, NY, NW} end

state_dim(::AbstractMeasurementModel{NX, NY, NW}) where {NX, NY, NW} = NX
measurement_dim(::AbstractMeasurementModel{NX, NY, NW}) where {NX, NY, NW} = NY
measurement_noise_dim(::AbstractMeasurementModel{NX, NY, NW}) where {NX, NY, NW} = NW

"""
    measure(model::AbstractMeasurementModel, x)

Compute the expected measurement for state x.

Returns: y = h(x)
"""
function measure end

"""
    jacobian_x(model::AbstractMeasurementModel, x)

Compute the Jacobian of the measurement function with respect to state.

Returns: H = ∂h/∂x evaluated at x
"""
function jacobian_x end

"""
    jacobian_w(model::AbstractMeasurementModel, x)

Compute the Jacobian of the measurement with respect to measurement noise.

Returns: M = ∂h/∂w evaluated at x

Default implementation assumes additive noise: M = I
"""
function jacobian_w(model::AbstractMeasurementModel{NX, NY, NW}, x) where {NX, NY, NW}
    return SMatrix{NY, NW, Float64}(I(NY))
end

"""
    measurement_noise_cov(model::AbstractMeasurementModel)

Return the measurement noise covariance matrix R.
"""
function measurement_noise_cov end

"""
    log_likelihood(model::AbstractMeasurementModel, y, x)

Log of the measurement likelihood, log p(y|x), for a given state.

Default implementation assumes additive Gaussian measurement noise. Add a method
for your model type to use a different noise distribution. Particle filters weight
with this rather than with `likelihood`, so that a measurement far out in the tail
of every particle still reweights them instead of underflowing to zero.
"""
function log_likelihood(model::AbstractMeasurementModel, y, x)
    y_pred = measure(model, x)
    R = measurement_noise_cov(model)
    ν = y - y_pred

    # Gaussian: log p = -0.5 (n log 2π + log|R| + ν'R⁻¹ν)
    n = length(y)
    return -0.5 * (n * log(2π) + logdet(R) + dot(ν, R \ ν))
end

"""
    likelihood(model::AbstractMeasurementModel, y, x)

Measurement likelihood p(y|x) for a given state, `exp(log_likelihood(model, y, x))`.
This underflows to zero for large residuals; use `log_likelihood` for weighting.
"""
likelihood(model::AbstractMeasurementModel, y, x) = exp(log_likelihood(model, y, x))
