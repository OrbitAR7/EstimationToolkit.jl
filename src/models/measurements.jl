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
    likelihood(model::AbstractMeasurementModel, y, x)

Compute the measurement likelihood p(y|x) for a given state.

Default implementation assumes Gaussian measurement noise.
"""
function likelihood(model::AbstractMeasurementModel, y, x)
    y_pred = measure(model, x)
    R = measurement_noise_cov(model)
    ν = y - y_pred
    
    # Gaussian likelihood: (2π)^(-n/2) |R|^(-1/2) exp(-0.5 ν'R⁻¹ν)
    n = length(y)
    log_det = logdet(R)
    exponent = -0.5 * dot(ν, R \ ν)
    log_likelihood = -0.5 * (n * log(2π) + log_det) + exponent
    
    return exp(log_likelihood)
end
