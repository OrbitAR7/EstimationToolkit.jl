"""
    UnscentedKalmanFilter{T<:Real, N}

Unscented Kalman filter for nonlinear state estimation.

The UKF uses the unscented transform to propagate mean and covariance through
nonlinear transformations without requiring Jacobian calculations. It typically
achieves better accuracy than the EKF for highly nonlinear systems.

# Fields
- `x::SVector{N, T}`: Current state estimate
- `P::SMatrix{N, N, T}`: Current error covariance
- `α::T`: Spread parameter (default 1e-3)
- `β::T`: Distribution parameter (default 2.0 for Gaussian)
- `κ::T`: Secondary scaling (default 0.0)
- `ν::Union{Nothing, SVector}`: Innovation from last update
- `S::Union{Nothing, SMatrix}`: Innovation covariance from last update

# References
- EstimationNotes.pdf, Chapter 15
- Julier & Uhlmann, "Unscented Filtering and Nonlinear Estimation"
"""
mutable struct UnscentedKalmanFilter{T<:Real, N} <: AbstractFilter
    x::SVector{N, T}
    P::SMatrix{N, N, T}
    
    # UKF parameters
    α::T
    β::T
    κ::T
    
    # Innovation statistics
    ν::Union{Nothing, SVector}
    S::Union{Nothing, SMatrix}
    
    function UnscentedKalmanFilter(x0::SVector{N, T}, P0::SMatrix{N, N, T};
                                   α=1e-3, β=2.0, κ=0.0) where {N, T<:Real}
        @assert size(P0) == (N, N) "Covariance must be N×N"
        @assert is_positive_definite(P0) "Initial covariance must be SPD"
        @assert α > 0 "α must be positive"
        
        new{T, N}(x0, P0, α, β, κ, nothing, nothing)
    end
end

# Convenience constructors
function UnscentedKalmanFilter(x0::AbstractVector{T}, P0::AbstractMatrix{T};
                               α=1e-3, β=2.0, κ=0.0) where {T<:Real}
    N = length(x0)
    return UnscentedKalmanFilter(SVector{N, T}(x0), SMatrix{N, N, T}(P0);
                                 α=α, β=β, κ=κ)
end

"""
    predict!(ukf::UnscentedKalmanFilter, model::AbstractDynamicsModel, u, dt)

UKF prediction step using the unscented transform.

Propagates sigma points through the nonlinear dynamics and reconstructs
the predicted mean and covariance.

Algorithm:
1. Generate sigma points from current state
2. Propagate each sigma point through dynamics
3. Reconstruct predicted mean and covariance
4. Add process noise
"""
function predict!(ukf::UnscentedKalmanFilter{T, N}, model::AbstractDynamicsModel,
                  u, dt) where {T, N}
    # Generate sigma points
    sigma = unscented_transform(ukf.x, ukf.P; α=ukf.α, β=ukf.β, κ=ukf.κ)
    
    # Propagate sigma points through dynamics
    propagate_fn = χ -> propagate(model, χ, u, dt)
    χ_pred = transform_sigma_points(propagate_fn, sigma)
    
    # Reconstruct mean and covariance
    x̄, P̄ = reconstruct_gaussian(χ_pred, sigma.weights_mean, sigma.weights_cov)
    
    # Add process noise
    Q = process_noise_cov(model, dt)
    P̄ = P̄ + Q
    P̄ = make_symmetric(P̄)
    
    ukf.x = SVector{N, T}(x̄)
    ukf.P = SMatrix{N, N, T}(P̄)
    
    return ukf
end

"""
    update!(ukf::UnscentedKalmanFilter, model::AbstractMeasurementModel, y)

UKF measurement update using the unscented transform.

Algorithm:
1. Generate sigma points from predicted state
2. Transform sigma points through measurement model
3. Compute predicted measurement and innovation covariance
4. Compute cross-covariance between state and measurement
5. Update state and covariance
"""
function update!(ukf::UnscentedKalmanFilter{T, N}, model::AbstractMeasurementModel,
                 y) where {T, N}
    # Generate sigma points from predicted state
    sigma = unscented_transform(ukf.x, ukf.P; α=ukf.α, β=ukf.β, κ=ukf.κ)
    
    # Transform sigma points through measurement model
    measure_fn = χ -> measure(model, χ)
    γ = transform_sigma_points(measure_fn, sigma)
    
    # Predicted measurement
    ŷ, Pyy = reconstruct_gaussian(γ, sigma.weights_mean, sigma.weights_cov)
    
    # Add measurement noise
    R = measurement_noise_cov(model)
    Pyy = Pyy + R
    Pyy = make_symmetric(Pyy)
    
    # Cross-covariance between state and measurement
    ny = length(ŷ)
    Pxy = zeros(T, N, ny)
    for (i, (χ, γi)) in enumerate(zip(sigma.points, γ))
        dx = χ - ukf.x
        dy = γi - ŷ
        Pxy += sigma.weights_cov[i] * (dx * dy')
    end
    
    # Kalman gain
    K = Pxy * inv(Pyy)
    
    # Innovation
    ν = y - ŷ
    
    # State update
    x = ukf.x + K * ν
    
    # Covariance update
    P = ukf.P - K * Pyy * K'
    P = make_symmetric(P)
    P = ensure_spd(P)  # Sometimes numerical errors make P slightly non-SPD
    
    # Store innovation statistics
    ukf.ν = ν
    ukf.S = Pyy
    
    ukf.x = SVector{N, T}(x)
    ukf.P = SMatrix{N, N, T}(P)
    
    return ukf
end

# Interface implementations
state_estimate(ukf::UnscentedKalmanFilter) = ukf.x
covariance(ukf::UnscentedKalmanFilter) = ukf.P
innovation(ukf::UnscentedKalmanFilter) = ukf.ν
innovation_covariance(ukf::UnscentedKalmanFilter) = ukf.S

UnscentedKalmanFilter(belief::GaussianBelief{T, N}; kwargs...) where {T, N} = 
    UnscentedKalmanFilter(belief.μ, belief.Σ; kwargs...)

GaussianBelief(ukf::UnscentedKalmanFilter) = GaussianBelief(ukf.x, ukf.P)
