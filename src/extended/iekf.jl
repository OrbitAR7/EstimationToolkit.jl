"""
    IteratedExtendedKalmanFilter{T<:Real, N}

Iterated Extended Kalman Filter (IEKF) for nonlinear state estimation.

The IEKF improves upon the standard EKF by performing multiple Gauss-Newton
iterations during the measurement update step. This refines the linearization
point, making the filter more accurate for highly nonlinear measurement models.

Instead of a single linearization about x̄ (prior), the IEKF iterates:
1. Linearize about x̂⁽ⁱ⁾
2. Compute Kalman gain and update
3. Check convergence
4. If not converged, return to step 1 with new x̂⁽ⁱ⁺¹⁾

This is equivalent to solving the nonlinear least-squares problem:
    min (x - x̄)' P̄⁻¹ (x - x̄) + (y - h(x))' R⁻¹ (y - h(x))

using Gauss-Newton iterations.

# Fields
- `x::SVector{N, T}`: Current state estimate
- `P::SMatrix{N, N, T}`: Current error covariance
- `max_iterations::Int`: Maximum number of iterations (default: 5)
- `tolerance::T`: Convergence tolerance on state change (default: 1e-6)
- `ν::Union{Nothing, SVector}`: Innovation from last update
- `S::Union{Nothing, SMatrix}`: Innovation covariance from last update
- `num_iterations::Int`: Number of iterations used in last update

# References
- EstimationNotes.pdf, Chapter 13, §13.4.1 "Iterated Extended Kalman Filter"
- 03-Nonlinear_filters.pdf, §2.1.1
- Bell, B.M., and Cathey, F.W. (1993). "The iterated Kalman filter update as a
  Gauss-Newton method", IEEE Trans. Automatic Control

# Notes
The IEKF is particularly useful when:
- Measurement model h(x) is highly nonlinear
- Innovation is large (linearization about prior is poor)
- Higher accuracy is needed without switching to particle methods

For linear measurements, IEKF converges in one iteration (equivalent to EKF).
"""
mutable struct IteratedExtendedKalmanFilter{T<:Real, N} <: AbstractFilter
    x::SVector{N, T}
    P::SMatrix{N, N, T}
    
    # Iteration parameters
    max_iterations::Int
    tolerance::T
    
    # Diagnostics
    ν::Union{Nothing, SVector}
    S::Union{Nothing, SMatrix}
    num_iterations::Int
    
    function IteratedExtendedKalmanFilter(x0::SVector{N, T}, P0::SMatrix{N, N, T};
                                         max_iterations::Int=5, 
                                         tolerance::T=T(1e-6)) where {N, T<:Real}
        @assert size(P0) == (N, N) "Covariance must be N×N"
        @assert is_positive_definite(P0) "Initial covariance must be SPD"
        @assert max_iterations > 0 "max_iterations must be positive"
        @assert tolerance > 0 "tolerance must be positive"
        
        new{T, N}(x0, P0, max_iterations, tolerance, nothing, nothing, 0)
    end
end

# Convenience constructors
function IteratedExtendedKalmanFilter(x0::AbstractVector{T}, P0::AbstractMatrix{T};
                                     max_iterations::Int=5,
                                     tolerance::T=T(1e-6)) where {T<:Real}
    N = length(x0)
    return IteratedExtendedKalmanFilter(SVector{N, T}(x0), SMatrix{N, N, T}(P0);
                                       max_iterations=max_iterations,
                                       tolerance=tolerance)
end

"""
    predict!(iekf::IteratedExtendedKalmanFilter, model::AbstractDynamicsModel, u, dt)

IEKF prediction step (identical to standard EKF).

The prediction uses single-point linearization since the dynamics are typically
less problematic than measurements. For highly nonlinear dynamics, consider
using UKF or particle methods instead.
"""
function predict!(iekf::IteratedExtendedKalmanFilter{T, N}, model::AbstractDynamicsModel,
                  u, dt) where {T, N}
    # Linearize dynamics about current estimate
    F = jacobian_x(model, iekf.x, u, dt)
    Q = process_noise_cov(model, dt)
    
    # State propagation using full nonlinear model
    x̄ = propagate(model, iekf.x, u, dt)
    
    # Covariance propagation using linearized model
    P̄ = F * iekf.P * F' + Q
    P̄ = make_symmetric(P̄)
    
    iekf.x = x̄
    iekf.P = SMatrix{N, N, T}(P̄)
    
    return iekf
end

"""
    update!(iekf::IteratedExtendedKalmanFilter, model::AbstractMeasurementModel, y)

IEKF measurement update with Gauss-Newton iterations.

Algorithm:
1. Initialize: x̂⁽⁰⁾ = x̄ (prior)
2. For i = 1, 2, ..., max_iterations:
   a. Linearize: H⁽ⁱ⁾ = ∂h/∂x |_{x̂⁽ⁱ⁻¹⁾}
   b. Compute innovation: ν⁽ⁱ⁾ = y - h(x̂⁽ⁱ⁻¹⁾)
   c. Compute gain: K⁽ⁱ⁾ = P̄ H⁽ⁱ⁾' (H⁽ⁱ⁾ P̄ H⁽ⁱ⁾' + R)⁻¹
   d. Update state: x̂⁽ⁱ⁾ = x̄ + K⁽ⁱ⁾ (ν⁽ⁱ⁾ + H⁽ⁱ⁾ (x̂⁽ⁱ⁻¹⁾ - x̄))
   e. Check convergence: ||x̂⁽ⁱ⁾ - x̂⁽ⁱ⁻¹⁾|| < tolerance
3. Final covariance: P = (I - K⁽ⁱ⁾ H⁽ⁱ⁾) P̄

# Convergence
The iteration typically converges in 2-4 iterations for moderately nonlinear
problems. If max_iterations is reached without convergence, the filter still
returns the last iterate (which is often usable).
"""
function update!(iekf::IteratedExtendedKalmanFilter{T, N}, model::AbstractMeasurementModel, y) where {T, N}
    # Store prior for iterations
    x̄ = iekf.x
    P̄ = iekf.P
    
    R = measurement_noise_cov(model)
    
    # Initialize iteration
    x̂ = x̄
    converged = false
    iter = 0
    
    H = nothing  # Will store final Jacobian
    S = nothing  # Will store final innovation covariance
    ν = nothing  # Will store final innovation
    
    for i in 1:iekf.max_iterations
        iter = i
        
        # Linearize measurement model about current iterate
        H = jacobian_x(model, x̂)
        
        # Predicted measurement at current iterate
        ŷ = measure(model, x̂)
        
        # Innovation
        ν = y - ŷ
        
        # Innovation covariance
        S = H * P̄ * H' + R
        S = make_symmetric(S)
        
        # Kalman gain
        K = P̄ * H' / S
        
        # Gauss-Newton update
        # x̂_new = x̄ + K * (ν + H * (x̂ - x̄))
        # This form ensures we're minimizing the cost function
        dx = K * (ν + H * (x̂ - x̄))
        x̂_new = x̄ + dx
        
        # Check convergence
        if norm(x̂_new - x̂) < iekf.tolerance
            x̂ = x̂_new
            converged = true
            break
        end
        
        x̂ = x̂_new
    end
    
    # Final covariance update using Joseph form
    # Use final Jacobian H
    K = P̄ * H' / S
    I_KH = SMatrix{N,N,T}(I) - K * H
    P = I_KH * P̄ * I_KH' + K * R * K'
    P = make_symmetric(P)
    
    # Store diagnostics
    iekf.ν = ν
    iekf.S = S
    iekf.num_iterations = iter
    
    # Update state and covariance
    iekf.x = x̂
    iekf.P = SMatrix{N, N, T}(P)
    
    return iekf
end

"""
    step!(iekf::IteratedExtendedKalmanFilter, dyn, meas, u, y, dt)

Combined prediction and update step.
"""
function step!(iekf::IteratedExtendedKalmanFilter, dyn::AbstractDynamicsModel,
               meas::AbstractMeasurementModel, u, y, dt)
    predict!(iekf, dyn, u, dt)
    update!(iekf, meas, y)
    return iekf
end

"""
    state_estimate(iekf::IteratedExtendedKalmanFilter)

Return the current state estimate.
"""
function state_estimate(iekf::IteratedExtendedKalmanFilter)
    return iekf.x
end

"""
    covariance(iekf::IteratedExtendedKalmanFilter)

Return the current error covariance.
"""
function covariance(iekf::IteratedExtendedKalmanFilter)
    return iekf.P
end

"""
    innovation(iekf::IteratedExtendedKalmanFilter)

Return the innovation from the last update.
"""
function innovation(iekf::IteratedExtendedKalmanFilter)
    return iekf.ν
end

"""
    innovation_covariance(iekf::IteratedExtendedKalmanFilter)

Return the innovation covariance from the last update.
"""
function innovation_covariance(iekf::IteratedExtendedKalmanFilter)
    return iekf.S
end

"""
    normalized_innovation_squared(iekf::IteratedExtendedKalmanFilter)

Compute NIS statistic: ν' S⁻¹ ν
"""
function normalized_innovation_squared(iekf::IteratedExtendedKalmanFilter)
    if isnothing(iekf.ν) || isnothing(iekf.S)
        return nothing
    end
    
    return iekf.ν' * (iekf.S \ iekf.ν)
end

"""
    iterations_used(iekf::IteratedExtendedKalmanFilter)

Return the number of iterations used in the last update.
"""
function iterations_used(iekf::IteratedExtendedKalmanFilter)
    return iekf.num_iterations
end
