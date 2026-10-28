"""
    ExtendedKalmanFilter{T<:Real, N}

Extended Kalman filter for nonlinear state estimation.

The EKF linearizes nonlinear dynamics and measurements about the current
state estimate using first-order Taylor expansion.

System model:
- Dynamics: x_{k+1} = f(x_k, u_k) + v_k,  v_k ~ N(0, Q_k)
- Measurement: y_k = h(x_k) + w_k,  w_k ~ N(0, R_k)

Linearization:
- F_k = ∂f/∂x |_{x̂_k}
- H_k = ∂h/∂x |_{x̄_k}

# Fields
- `x::SVector{N, T}`: Current state estimate
- `P::SMatrix{N, N, T}`: Current error covariance
- `ν::Union{Nothing, SVector}`: Innovation from last update
- `S::Union{Nothing, SMatrix}`: Innovation covariance from last update

# References
- T. Humphreys, Statistical Estimation Theory course notes (UT Austin ASE 381P.6, 2023), §13.3
- R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 4: Linear Filters (UT Austin ASE 381P.7, 2023), §2
"""
mutable struct ExtendedKalmanFilter{T<:Real, N} <: AbstractFilter
    x::SVector{N, T}
    P::SMatrix{N, N, T}
    
    ν::Union{Nothing, SVector}
    S::Union{Nothing, SMatrix}
    
    function ExtendedKalmanFilter(x0::SVector{N, T}, P0::SMatrix{N, N, T}) where {N, T<:Real}
        @assert size(P0) == (N, N) "Covariance must be N×N"
        @assert is_positive_definite(P0) "Initial covariance must be SPD"
        
        new{T, N}(x0, P0, nothing, nothing)
    end
end

# Convenience constructors
function ExtendedKalmanFilter(x0::AbstractVector{T}, P0::AbstractMatrix{T}) where {T<:Real}
    N = length(x0)
    return ExtendedKalmanFilter(SVector{N, T}(x0), SMatrix{N, N, T}(P0))
end

"""
    predict!(ekf::ExtendedKalmanFilter, model::AbstractDynamicsModel, u, dt)

EKF prediction step with first-order linearization.

Propagates state using the full nonlinear model, but propagates covariance
using linearization:
- x̄ = f(x, u)
- P̄ = F P F' + Q, where F = ∂f/∂x |_x
"""
function predict!(ekf::ExtendedKalmanFilter{T, N}, model::AbstractDynamicsModel, 
                  u, dt) where {T, N}
    # Linearize dynamics about current estimate
    F = jacobian_x(model, ekf.x, u, dt)
    Q = process_noise_cov(model, dt)
    
    # State propagation using full nonlinear model
    x̄ = propagate(model, ekf.x, u, dt)
    
    # Covariance propagation using linearized model
    P̄ = F * ekf.P * F' + Q
    P̄ = make_symmetric(P̄)
    
    ekf.x = x̄
    ekf.P = SMatrix{N, N, T}(P̄)
    
    return ekf
end

"""
    update!(ekf::ExtendedKalmanFilter, model::AbstractMeasurementModel, y)

EKF measurement update with linearization.

Linearizes measurement model about predicted state:
- ν = y - h(x̄)
- S = H P̄ H' + R, where H = ∂h/∂x |_{x̄}
- K = P̄ H' S⁻¹
- x = x̄ + K ν
- P = (I - K H) P̄
"""
function update!(ekf::ExtendedKalmanFilter{T, N}, model::AbstractMeasurementModel, 
                 y) where {T, N}
    # Linearize measurement model about predicted state
    H = jacobian_x(model, ekf.x)
    R = measurement_noise_cov(model)
    
    # Predicted measurement using full nonlinear model
    ŷ = measure(model, ekf.x)
    
    # Innovation
    ν = y - ŷ
    
    # Innovation covariance
    S = H * ekf.P * H' + R
    S = make_symmetric(S)
    
    # Kalman gain
    K = ekf.P * H' * inv(S)
    
    # State update
    x = ekf.x + K * ν
    
    # Covariance update (Joseph form for stability)
    P = joseph_form_update(ekf.P, K, H, R)
    P = make_symmetric(P)
    
    # Store innovation statistics
    ekf.ν = ν
    ekf.S = S
    
    ekf.x = x
    ekf.P = SMatrix{N, N, T}(P)
    
    return ekf
end

# Interface implementations
state_estimate(ekf::ExtendedKalmanFilter) = ekf.x
covariance(ekf::ExtendedKalmanFilter) = ekf.P
innovation(ekf::ExtendedKalmanFilter) = ekf.ν
innovation_covariance(ekf::ExtendedKalmanFilter) = ekf.S

ExtendedKalmanFilter(belief::GaussianBelief{T, N}) where {T, N} = 
    ExtendedKalmanFilter(belief.μ, belief.Σ)

GaussianBelief(ekf::ExtendedKalmanFilter) = GaussianBelief(ekf.x, ekf.P)

"""
    get_predicted_innovation_stats(ekf::ExtendedKalmanFilter, model, y)

Innovation and its covariance at the current (predicted) state, without updating.
Lets the EKF be used inside IMM/GPB filter banks.
"""
function get_predicted_innovation_stats(ekf::ExtendedKalmanFilter, model::AbstractMeasurementModel, y)
    H = jacobian_x(model, ekf.x)
    R = measurement_noise_cov(model)
    ν = y - measure(model, ekf.x)
    S = H * ekf.P * H' + R
    return ν, make_symmetric(S)
end
