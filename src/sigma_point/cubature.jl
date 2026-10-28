"""
    CubatureKalmanFilter{T<:Real, N}

Cubature Kalman filter using spherical-radial cubature rule for sigma points.

The CKF uses the spherical cubature rule to generate 2n sigma points (vs 2n+1 for UKF)
with equal weights. This provides:
- Third-order accuracy for Gaussian inputs
- Minimal number of sigma points
- No tuning parameters (unlike UKF's α, β, κ)
- Deterministic sampling without randomness

The cubature rule integrates polynomials up to order 3 exactly, making it more
accurate than the first-order linearization of EKF while maintaining computational
efficiency.

# Algorithm
**Prediction:**
1. Generate 2n cubature points: χᵢ = x̄ ± √n S_x (columns)
2. Propagate: χᵢ⁺ = f(χᵢ, u)  
3. Reconstruct: x̄⁺ = (1/2n) Σᵢ χᵢ⁺,  P̄⁺ = (1/2n) Σᵢ (χᵢ⁺ - x̄⁺)(χᵢ⁺ - x̄⁺)' + Q

**Update:**
1. Generate 2n cubature points from prediction: χᵢ = x̄ ± √n S_P̄
2. Predict measurements: γᵢ = h(χᵢ)
3. Reconstruct: ŷ = (1/2n) Σᵢ γᵢ,  P_yy = (1/2n) Σᵢ (γᵢ - ŷ)(γᵢ - ŷ)' + R
4. Cross-covariance: P_xy = (1/2n) Σᵢ (χᵢ - x̄)(γᵢ - ŷ)'
5. Update: K = P_xy P_yy⁻¹,  x⁺ = x̄ + K(y - ŷ),  P⁺ = P̄ - K P_yy K'

# Fields
- `x::SVector{N, T}`: Current state estimate
- `P::SMatrix{N, N, T}`: Current error covariance
- `ν::Union{Nothing, SVector}`: Innovation from last update
- `S::Union{Nothing, SMatrix}`: Innovation covariance from last update

# References
- R. Zanetti, "Quadrature Kalman Filter," ASE 381P.7 lecture slides (UT Austin, 2023)
- R. Zanetti, "Other Kalman Filters," ASE 381P.7 lecture slides (UT Austin, 2023)
- Arasaratnam, I., & Haykin, S. (2009). "Cubature Kalman Filters",
  IEEE Transactions on Automatic Control
- T. Humphreys, Statistical Estimation Theory course notes (UT Austin ASE 381P.6, 2023), Ch. 15 (UKF, for comparison)

# Comparison with UKF
- CKF: 2n points, no tuning, 3rd order accuracy
- UKF: 2n+1 points, requires α/β/κ tuning, 3rd order accuracy
- CKF generally preferred for its simplicity and robustness
"""
mutable struct CubatureKalmanFilter{T<:Real, N} <: AbstractFilter
    x::SVector{N, T}
    P::SMatrix{N, N, T}
    
    # Store innovation statistics for diagnostics
    ν::Union{Nothing, SVector}
    S::Union{Nothing, SMatrix}
    
    function CubatureKalmanFilter(x0::SVector{N, T}, P0::SMatrix{N, N, T}) where {N, T<:Real}
        @assert size(P0) == (N, N) "Covariance must be N×N"
        @assert is_positive_definite(P0) "Initial covariance must be SPD"
        
        new{T, N}(x0, P0, nothing, nothing)
    end
end

# Convenience constructors
function CubatureKalmanFilter(x0::AbstractVector{T}, P0::AbstractMatrix{T}) where {T<:Real}
    N = length(x0)
    return CubatureKalmanFilter(SVector{N, T}(x0), SMatrix{N, N, T}(P0))
end

"""
    predict!(ckf::CubatureKalmanFilter, model::AbstractDynamicsModel, u, dt)

Cubature Kalman filter prediction step.

Uses spherical cubature rule to propagate the state distribution through
nonlinear dynamics without linearization.

# Algorithm
1. Generate 2n cubature points via cubature_transform
2. Propagate each point through f(χᵢ, u)
3. Reconstruct mean and covariance from propagated points
4. Add process noise Q
"""
function predict!(ckf::CubatureKalmanFilter{T, N}, model::AbstractDynamicsModel, 
                  u, dt) where {T, N}
    # Generate cubature points
    sigma_points = cubature_transform(ckf.x, ckf.P)
    
    # Propagate sigma points through dynamics
    f_wrap = χ -> propagate(model, χ, u, dt)
    χ_prop = transform_sigma_points(f_wrap, sigma_points)
    
    # Reconstruct Gaussian from propagated points
    x̄, P̄_xx = reconstruct_gaussian(χ_prop, sigma_points.weights_mean, 
                                     sigma_points.weights_cov)
    
    # Add process noise
    Q = process_noise_cov(model, dt)
    P̄ = P̄_xx + Q
    P̄ = make_symmetric(P̄)
    
    # Convert to static arrays
    ckf.x = SVector{N, T}(x̄)
    ckf.P = SMatrix{N, N, T}(P̄)
    
    return ckf
end

"""
    update!(ckf::CubatureKalmanFilter, model::AbstractMeasurementModel, y)

Cubature Kalman filter measurement update.

Uses cubature points to propagate the predicted distribution through the
measurement model, computing the cross-covariance exactly (up to cubature accuracy).

# Algorithm
1. Generate 2n cubature points from prediction
2. Transform through measurement model h(χᵢ)
3. Compute predicted measurement mean and covariance
4. Compute cross-covariance between state and measurement
5. Standard Kalman update with these statistics
"""
function update!(ckf::CubatureKalmanFilter{T, N}, model::AbstractMeasurementModel, y) where {T, N}
    # Generate cubature points from predicted distribution
    sigma_points = cubature_transform(ckf.x, ckf.P)
    
    # Propagate through measurement model
    h_wrap = χ -> measure(model, χ)
    γ = transform_sigma_points(h_wrap, sigma_points)
    
    # Predicted measurement statistics
    ŷ, P_yy_clean = reconstruct_gaussian(γ, sigma_points.weights_mean,
                                         sigma_points.weights_cov)
    
    # Add measurement noise
    R = measurement_noise_cov(model)
    P_yy = P_yy_clean + R
    P_yy = make_symmetric(P_yy)
    
    # Compute cross-covariance P_xy
    n_y = length(ŷ)
    P_xy = zeros(T, N, n_y)
    
    for (w, χᵢ, γᵢ) in zip(sigma_points.weights_cov, sigma_points.points, γ)
        dx = χᵢ - ckf.x
        dy = γᵢ - ŷ
        P_xy += w * (dx * dy')
    end
    
    # Innovation
    ν = y - ŷ
    
    # Kalman gain
    K = P_xy / P_yy
    
    # State update
    x⁺ = ckf.x + K * ν
    
    # Covariance update (Joseph form for numerical stability)
    I_KH_approx = SMatrix{N,N,T}(I) - K * (P_xy' / ckf.P)  # Approximate I - KH
    P⁺ = ckf.P - K * P_yy * K'
    P⁺ = make_symmetric(P⁺)
    
    # Ensure SPD
    P⁺ = ensure_spd(P⁺)
    
    # Store innovation statistics
    ckf.ν = SVector{n_y, T}(ν)
    ckf.S = SMatrix{n_y, n_y, T}(P_yy)
    
    # Update filter state
    ckf.x = SVector{N, T}(x⁺)
    ckf.P = SMatrix{N, N, T}(P⁺)
    
    return ckf
end

"""
    step!(ckf::CubatureKalmanFilter, dyn, meas, u, y, dt)

Combined prediction and update step.
"""
function step!(ckf::CubatureKalmanFilter, dyn::AbstractDynamicsModel,
               meas::AbstractMeasurementModel, u, y, dt)
    predict!(ckf, dyn, u, dt)
    update!(ckf, meas, y)
    return ckf
end

"""
    state_estimate(ckf::CubatureKalmanFilter)

Return the current state estimate.
"""
function state_estimate(ckf::CubatureKalmanFilter)
    return ckf.x
end

"""
    covariance(ckf::CubatureKalmanFilter)

Return the current error covariance.
"""
function covariance(ckf::CubatureKalmanFilter)
    return ckf.P
end

"""
    innovation(ckf::CubatureKalmanFilter)

Return the innovation (measurement residual) from the last update.
"""
function innovation(ckf::CubatureKalmanFilter)
    return ckf.ν
end

"""
    innovation_covariance(ckf::CubatureKalmanFilter)

Return the innovation covariance from the last update.
"""
function innovation_covariance(ckf::CubatureKalmanFilter)
    return ckf.S
end

"""
    normalized_innovation_squared(ckf::CubatureKalmanFilter)

Compute NIS statistic: ν' S⁻¹ ν
"""
function normalized_innovation_squared(ckf::CubatureKalmanFilter)
    if isnothing(ckf.ν) || isnothing(ckf.S)
        return nothing
    end
    
    return ckf.ν' * (ckf.S \ ckf.ν)
end
