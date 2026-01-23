"""
    SecondOrderExtendedKalmanFilter

Second-Order Extended Kalman Filter (SOEKF) including Hessian terms.

The SOEKF improves upon the standard EKF by including second-order (Hessian) terms
in the Taylor series expansion. This provides better accuracy for highly nonlinear
systems at the cost of additional computation for second derivatives.

# Mathematical Formulation

**Standard EKF** uses first-order Taylor expansion:
```
f(x) ≈ f(x̄) + F·(x - x̄)  where F = ∂f/∂x
```

**Second-Order EKF** uses second-order Taylor expansion:
```
f(x) ≈ f(x̄) + F·(x - x̄) + 0.5·(x - x̄)'·H·(x - x̄)
```
where H is the Hessian matrix: Hᵢⱼₖ = ∂²fᵢ/∂xⱼ∂xₖ

**Covariance Propagation** includes Hessian correction:
```
P̄ ≈ F·P·F' + Q + 0.5·Tr(H·P·H'·P)
```

**Reference**: Anderson & Moore (1979), "Optimal Filtering"
**Lecture Source**: Lecture23-OtherKF.pdf, EstimationNotes §13.5

# Fields
- `x::SVector{N,T}` - State estimate
- `P::SMatrix{N,N,T}` - Covariance matrix
- `ν::Union{Nothing, SVector}` - Innovation
- `S::Union{Nothing, SMatrix}` - Innovation covariance

# Constructor
```julia
SecondOrderExtendedKalmanFilter(x0::SVector{N,T}, P0::SMatrix{N,N,T})
```

# Example
```julia
# Initialize filter
soekf = SecondOrderExtendedKalmanFilter(x0, P0)

# Filtering cycle
for (u, y) in zip(controls, measurements)
    predict!(soekf, dynamics_model, u, dt)
    update!(soekf, measurement_model, y)
end
```

# Performance Characteristics
- **Time complexity**: O(n³) due to Hessian computations
- **Accuracy**: Better than EKF for highly nonlinear systems
- **Computational cost**: ~2-3x standard EKF
- **Hessian computation**: Via automatic differentiation (ForwardDiff.jl)
- **Best use**: Highly nonlinear dynamics where EKF accuracy is insufficient
"""
mutable struct SecondOrderExtendedKalmanFilter{T<:Real, N} <: AbstractFilter
    x::SVector{N, T}
    P::SMatrix{N, N, T}
    ν::Union{Nothing, SVector{<:Any, T}}
    S::Union{Nothing, SMatrix{<:Any, <:Any, T}}
    
    function SecondOrderExtendedKalmanFilter(x0::SVector{N,T}, P0::SMatrix{N,N,T}) where {N,T}
        # Ensure P0 is SPD
        P0_sym = make_symmetric(P0)
        P0_spd = ensure_spd(P0_sym)
        
        new{T,N}(x0, P0_spd, nothing, nothing)
    end
end

"""
    compute_hessian(f, x)

Compute Hessian matrix (second derivatives) of vector function f at point x.

Uses automatic differentiation via ForwardDiff.jl.

# Arguments
- `f`: Vector function ℝⁿ → ℝᵐ
- `x`: Evaluation point (n-dimensional)

# Returns
Array of Hessian matrices, one m×n×n array where [i, j, k] = ∂²fᵢ/∂xⱼ∂xₖ
"""
function compute_hessian(f, x::SVector{N,T}) where {N,T}
    # For a vector function f: ℝⁿ → ℝᵐ, we need Hessian for each component
    # Convert to regular vector for ForwardDiff
    x_vec = Vector(x)
    f_vec(xv) = Vector(f(SVector{N}(xv)))
    f_val = f_vec(x_vec)
    M = length(f_val)
    
    # Hessian for each output dimension
    hessians = Vector{Matrix{T}}(undef, M)
    
    for i in 1:M
        # Extract i-th component as scalar function
        fᵢ(xv) = f_vec(xv)[i]
        
        # Compute Hessian using ForwardDiff
        hessians[i] = ForwardDiff.hessian(fᵢ, x_vec)
    end
    
    return hessians
end

"""
    predict!(soekf::SecondOrderExtendedKalmanFilter, model::AbstractDynamicsModel, u, dt)

Second-order EKF prediction step with Hessian correction.

# Algorithm
1. Propagate state: x̄ = f(x)
2. Compute Jacobian: F = ∂f/∂x
3. Compute Hessian: H = ∂²f/∂x²
4. First-order covariance: P̄₁ = F·P·F' + Q
5. Second-order correction: P̄₂ = 0.5·Σᵢ tr(Hᵢ·P)·Hᵢ·P
6. Total prediction: P̄ = P̄₁ + P̄₂
"""
function predict!(soekf::SecondOrderExtendedKalmanFilter{T,N}, model::AbstractDynamicsModel, u, dt) where {T,N}
    # Propagate state mean
    x_pred = propagate(model, soekf.x, u, dt)
    
    # Compute Jacobian (first-order)
    F = jacobian_x(model, soekf.x, u, dt)
    Q = process_noise_cov(model, dt)
    
    # First-order covariance prediction
    P_pred_first = F * soekf.P * F' + Q
    
    # Compute Hessians (second-order)
    f_wrap = x -> propagate(model, x, u, dt)
    hessians = compute_hessian(f_wrap, soekf.x)
    
    # Second-order covariance correction
    # For each state dimension i: Σⱼₖ Hᵢⱼₖ·Pⱼⱼ·Hᵢⱼₖ·Pₖₖ (simplified)
    # More accurate: 0.5·Σᵢ [tr(Hᵢ·P)·(Hᵢ·P·Hᵢ')]
    P_pred_second = zeros(T, N, N)
    
    for i in 1:N
        Hi = SMatrix{N,N,T}(hessians[i])
        # Correction term: 0.5 * trace(Hi * P) * Hi * P
        trace_HiP = tr(Hi * soekf.P)
        P_pred_second += 0.5 * trace_HiP * Hi * soekf.P * Hi'
    end
    
    # Total predicted covariance
    P_pred = P_pred_first + SMatrix{N,N,T}(P_pred_second)
    
    # Ensure symmetry and positive definiteness
    P_pred_sym = make_symmetric(P_pred)
    soekf.P = ensure_spd(P_pred_sym)
    soekf.x = x_pred
    
    # Reset innovation statistics
    soekf.ν = nothing
    soekf.S = nothing
end

"""
    update!(soekf::SecondOrderExtendedKalmanFilter, model::AbstractMeasurementModel, y)

Second-order EKF measurement update with Hessian correction.

# Algorithm
1. Predicted measurement: ŷ = h(x̄)
2. Innovation: ν = y - ŷ
3. Compute Jacobian: H = ∂h/∂x
4. Compute Hessian: Hₕ = ∂²h/∂x²
5. Innovation covariance with Hessian correction: S = H·P̄·H' + R + correction
6. Kalman gain: K = P̄·H'·S⁻¹
7. State update: x = x̄ + K·ν
8. Covariance update: P = (I - K·H)·P̄·(I - K·H)' + K·R·K' (Joseph form)
"""
function update!(soekf::SecondOrderExtendedKalmanFilter{T,N}, model::AbstractMeasurementModel, y::SVector{NY,T}) where {T,N,NY}
    # Predicted measurement
    ŷ = measure(model, soekf.x)
    
    # Innovation
    soekf.ν = y - ŷ
    
    # Compute Jacobian (first-order)
    H = jacobian_x(model, soekf.x)
    R = measurement_noise_cov(model)
    
    # Innovation covariance (first-order)
    S_first = H * soekf.P * H' + R
    
    # Compute Hessian (second-order)
    h_wrap = x -> measure(model, x)
    hessians_h = compute_hessian(h_wrap, soekf.x)
    
    # Second-order correction to innovation covariance
    S_second = zeros(T, NY, NY)
    for i in 1:NY
        Hi_h = SMatrix{N,N,T}(hessians_h[i])
        trace_HiP = tr(Hi_h * soekf.P)
        S_second += 0.5 * trace_HiP * Hi_h * soekf.P * Hi_h'
    end
    
    # Total innovation covariance
    S = S_first + SMatrix{NY,NY,T}(S_second)
    S_sym = make_symmetric(S)
    soekf.S = ensure_spd(S_sym)
    
    # Kalman gain
    K = (soekf.P * H') / Matrix(soekf.S)
    K_static = SMatrix{N,NY,T}(K)
    
    # State update
    soekf.x = soekf.x + K_static * soekf.ν
    
    # Covariance update (Joseph form for numerical stability)
    I_KH = SMatrix{N,N,T}(I) - K_static * H
    P_updated = I_KH * soekf.P * I_KH' + K_static * R * K_static'
    
    P_updated_sym = make_symmetric(P_updated)
    soekf.P = ensure_spd(P_updated_sym)
end

"""
    state_estimate(soekf::SecondOrderExtendedKalmanFilter)

Return current state estimate.
"""
function state_estimate(soekf::SecondOrderExtendedKalmanFilter)
    return soekf.x
end

"""
    covariance(soekf::SecondOrderExtendedKalmanFilter)

Return current covariance estimate.
"""
function covariance(soekf::SecondOrderExtendedKalmanFilter)
    return soekf.P
end

"""
    innovation(soekf::SecondOrderExtendedKalmanFilter)

Return the most recent innovation.
"""
function innovation(soekf::SecondOrderExtendedKalmanFilter)
    return soekf.ν
end

"""
    innovation_covariance(soekf::SecondOrderExtendedKalmanFilter)

Return the most recent innovation covariance.
"""
function innovation_covariance(soekf::SecondOrderExtendedKalmanFilter)
    return soekf.S
end

"""
    nis(soekf::SecondOrderExtendedKalmanFilter)

Compute Normalized Innovation Squared (NIS) statistic.
"""
function nis(soekf::SecondOrderExtendedKalmanFilter)
    if soekf.ν === nothing || soekf.S === nothing
        return nothing
    end
    return dot(soekf.ν, soekf.S \ Vector(soekf.ν))
end
