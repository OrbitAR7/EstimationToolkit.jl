"""
    KalmanFilter{T<:Real, N}

Linear Kalman filter for optimal state estimation in linear-Gaussian systems.

The Kalman filter is the optimal MMSE estimator for systems of the form:
- Dynamics: x_{k+1} = F_k x_k + G_k u_k + v_k,  v_k ~ N(0, Q_k)
- Measurement: y_k = H_k x_k + w_k,  w_k ~ N(0, R_k)

# Fields
- `x::SVector{N, T}`: Current state estimate
- `P::SMatrix{N, N, T}`: Current error covariance
- `ν::Union{Nothing, SVector}`: Innovation from last update
- `S::Union{Nothing, SMatrix}`: Innovation covariance from last update

# References
- EstimationNotes.pdf, Chapter 9
- 04-linear_filters.pdf
"""
mutable struct KalmanFilter{T<:Real, N} <: AbstractFilter
    x::SVector{N, T}
    P::SMatrix{N, N, T}
    
    # Store innovation statistics for diagnostics
    ν::Union{Nothing, SVector}
    S::Union{Nothing, SMatrix}
    
    function KalmanFilter(x0::SVector{N, T}, P0::SMatrix{N, N, T}) where {N, T<:Real}
        @assert size(P0) == (N, N) "Covariance must be N×N"
        @assert is_positive_definite(P0) "Initial covariance must be SPD"
        
        new{T, N}(x0, P0, nothing, nothing)
    end
end

# Convenience constructors
function KalmanFilter(x0::AbstractVector{T}, P0::AbstractMatrix{T}) where {T<:Real}
    N = length(x0)
    return KalmanFilter(SVector{N, T}(x0), SMatrix{N, N, T}(P0))
end

"""
    predict!(kf::KalmanFilter, model::AbstractDynamicsModel, u, dt)

Kalman filter prediction step.

Propagates state and covariance forward in time:
- x̄ = F x + G u
- P̄ = F P F' + Q

where F = jacobian_x(model, x, u, dt) and Q = process_noise_cov(model, dt)
"""
function predict!(kf::KalmanFilter{T, N}, model::AbstractDynamicsModel, u, dt) where {T, N}
    # Get linearized dynamics
    F = jacobian_x(model, kf.x, u, dt)
    Q = process_noise_cov(model, dt)
    
    # State propagation
    x̄ = propagate(model, kf.x, u, dt)
    
    # Covariance propagation
    P̄ = F * kf.P * F' + Q
    
    # Ensure numerical stability
    P̄ = make_symmetric(P̄)
    
    kf.x = x̄
    kf.P = SMatrix{N, N, T}(P̄)
    
    return kf
end

"""
    update!(kf::KalmanFilter, model::AbstractMeasurementModel, y)

Kalman filter measurement update step.

Updates state estimate based on measurement:
- ν = y - H x̄  (innovation)
- S = H P̄ H' + R  (innovation covariance)
- K = P̄ H' S⁻¹  (Kalman gain)
- x = x̄ + K ν
- P = (I - K H) P̄  (or Joseph form for stability)
"""
function update!(kf::KalmanFilter{T, N}, model::AbstractMeasurementModel, y) where {T, N}
    # Get measurement model
    H = jacobian_x(model, kf.x)
    R = measurement_noise_cov(model)
    
    # Predicted measurement
    ŷ = measure(model, kf.x)
    
    # Innovation
    ν = y - ŷ
    
    # Innovation covariance
    S = H * kf.P * H' + R
    S = make_symmetric(S)
    
    # Kalman gain
    K = kf.P * H' * inv(S)
    
    # State update
    x = kf.x + K * ν
    
    # Covariance update using Joseph form for numerical stability
    P = joseph_form_update(kf.P, K, H, R)
    P = make_symmetric(P)
    
    # Store innovation statistics
    kf.ν = ν
    kf.S = S
    
    kf.x = x
    kf.P = SMatrix{N, N, T}(P)
    
    return kf
end

# Interface implementations
state_estimate(kf::KalmanFilter) = kf.x
covariance(kf::KalmanFilter) = kf.P
innovation(kf::KalmanFilter) = kf.ν
innovation_covariance(kf::KalmanFilter) = kf.S

"""
    get_predicted_innovation_stats(kf::KalmanFilter, model::AbstractMeasurementModel, y)

Compute innovation statistics (ν, S) without updating the filter state.
Used by multiple-model filters (IMM, etc.) to compute likelihoods.
"""
function get_predicted_innovation_stats(kf::KalmanFilter, model::AbstractMeasurementModel, y)
    # Get measurement model
    H = jacobian_x(model, kf.x)
    R = measurement_noise_cov(model)
    
    # Predicted measurement
    ŷ = measure(model, kf.x)
    
    # Innovation
    ν = y - ŷ
    
    # Innovation covariance
    S = H * kf.P * H' + R
    
    return ν, make_symmetric(S)
end

"""
    KalmanFilter(belief::GaussianBelief)

Construct a Kalman filter from a Gaussian belief state.
"""
KalmanFilter(belief::GaussianBelief{T, N}) where {T, N} = KalmanFilter(belief.μ, belief.Σ)

"""
    GaussianBelief(kf::KalmanFilter)

Extract the current belief state from a Kalman filter.
"""
GaussianBelief(kf::KalmanFilter) = GaussianBelief(kf.x, kf.P)
