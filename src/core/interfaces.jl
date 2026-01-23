"""
    AbstractFilter

Base type for all filters in the estimation toolkit.

All concrete filter types should implement:
- `predict!(filter, model, u, dt)` - Time update
- `update!(filter, model, y)` - Measurement update
- `state_estimate(filter)` - Current state estimate
- `covariance(filter)` - Current error covariance (if applicable)
"""
abstract type AbstractFilter end

"""
    predict!(filter::AbstractFilter, model::AbstractDynamicsModel, u, dt)

Perform the prediction (time update) step of the filter.

# Arguments
- `filter`: Filter object (modified in-place)
- `model`: Dynamics model
- `u`: Control input (can be `nothing` if no control)
- `dt`: Time step

Returns the filter for chaining operations.
"""
function predict! end

"""
    update!(filter::AbstractFilter, model::AbstractMeasurementModel, y)

Perform the measurement update (correction) step of the filter.

# Arguments
- `filter`: Filter object (modified in-place)
- `model`: Measurement model
- `y`: Measurement vector

Returns the filter for chaining operations.
"""
function update! end

"""
    step!(filter::AbstractFilter, dyn::AbstractDynamicsModel, 
          meas::AbstractMeasurementModel, u, y, dt)

Convenience function that performs predict! followed by update!

# Arguments
- `filter`: Filter object
- `dyn`: Dynamics model
- `meas`: Measurement model
- `u`: Control input
- `y`: Measurement
- `dt`: Time step

Returns the filter for chaining.
"""
function step!(filter::AbstractFilter, dyn, meas, u, y, dt)
    predict!(filter, dyn, u, dt)
    update!(filter, meas, y)
    return filter
end

"""
    state_estimate(filter::AbstractFilter)

Returns the current state estimate.
"""
function state_estimate end

"""
    covariance(filter::AbstractFilter)

Returns the current estimation error covariance (for Gaussian filters).
"""
function covariance end

# Alias for consistency with some literature
const state_covariance = covariance

"""
    innovation(filter::AbstractFilter)

Returns the innovation (measurement residual) from the last update.
"""
function innovation end

"""
    innovation_covariance(filter::AbstractFilter)

Returns the innovation covariance from the last update.
"""
function innovation_covariance end

# Marginalized Filter Interfaces
function propagate_nonlinear end
function linear_dynamics_matrix end
function process_noise_cov_nonlinear end
function process_noise_cov_linear end

function linear_measurement_matrix end
function nonlinear_measurement end
