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

With a Gaussian prior N(x̄, P), the second-order terms give, for each output
component i with Hessian Hᵢ (exact when f is quadratic):
```
E[fᵢ(x)]      ≈ fᵢ(x̄) + ½·tr(Hᵢ·P)
Cov[fᵢ, fⱼ]   ≈ (F·P·F')ᵢⱼ + ½·tr(Hᵢ·P·Hⱼ·P)
```
The prediction uses these with f = dynamics (plus Q), the update with f = measurement
(plus R) for the predicted measurement and the innovation covariance.

**Reference**: Anderson & Moore (1979), "Optimal Filtering"
**References**: R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 4: Linear Filters (UT Austin ASE 381P.7, 2023), §2; R. Zanetti, "Other Kalman Filters," ASE 381P.7 lecture slides (UT Austin, 2023)

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
4. Mean with second-order bias: x̄ᵢ = fᵢ(x) + ½·tr(Hᵢ·P)
5. Covariance: P̄ = F·P·F' + Q + C, Cᵢⱼ = ½·tr(Hᵢ·P·Hⱼ·P)
"""
function predict!(soekf::SecondOrderExtendedKalmanFilter{T,N}, model::AbstractDynamicsModel, u, dt) where {T,N}
    # Propagate state mean
    x_pred = propagate(model, soekf.x, u, dt)
    
    # Compute Jacobian (first-order)
    F = jacobian_x(model, soekf.x, u, dt)
    Q = process_noise_cov(model, soekf.x, dt)
    
    # First-order covariance prediction
    P_pred_first = F * soekf.P * F' + Q
    
    # Compute Hessians (second-order)
    f_wrap = x -> propagate(model, x, u, dt)
    hessians = compute_hessian(f_wrap, soekf.x)
    
    # Second-order terms: mean bias ½·tr(Hᵢ·P) and covariance ½·tr(Hᵢ·P·Hⱼ·P)
    bias, C = _second_order_terms(hessians, soekf.P)
    x_pred = x_pred + SVector{N,T}(bias)

    # Total predicted covariance
    P_pred = P_pred_first + SMatrix{N,N,T}(C)
    
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
1. Compute Jacobian H = ∂h/∂x and Hessians Hᵢ = ∂²hᵢ/∂x² at x̄
2. Predicted measurement: ŷᵢ = hᵢ(x̄) + ½·tr(Hᵢ·P̄)
3. Innovation: ν = y - ŷ
4. Innovation covariance: S = H·P̄·H' + R + C, Cᵢⱼ = ½·tr(Hᵢ·P̄·Hⱼ·P̄)
5. Kalman gain: K = P̄·H'·S⁻¹
6. State update: x = x̄ + K·ν
7. Covariance update: P = P̄ - K·S·K'

Works for any measurement dimension (earlier versions required it to equal the
state dimension).
"""
function update!(soekf::SecondOrderExtendedKalmanFilter{T,N}, model::AbstractMeasurementModel, y::SVector{NY,T}) where {T,N,NY}
    # Jacobian and Hessians of the measurement at the predicted state
    H = jacobian_x(model, soekf.x)
    R = measurement_noise_cov(model)
    h_wrap = x -> measure(model, x)
    hessians_h = compute_hessian(h_wrap, soekf.x)
    bias, C = _second_order_terms(hessians_h, soekf.P)

    # Predicted measurement with the second-order bias, and innovation
    ŷ = measure(model, soekf.x) + SVector{NY,T}(bias)
    soekf.ν = y - ŷ

    # Innovation covariance with the second-order term
    S = H * soekf.P * H' + R + SMatrix{NY,NY,T}(C)
    soekf.S = ensure_spd(make_symmetric(S))

    # Kalman gain, state and covariance update
    K = SMatrix{N,NY,T}((soekf.P * H') / Matrix(soekf.S))
    soekf.x = soekf.x + K * soekf.ν
    P_updated = soekf.P - K * soekf.S * K'
    soekf.P = ensure_spd(make_symmetric(P_updated))
    return soekf
end

"""
    _second_order_terms(hessians, P)

Second-order moment terms of a vector function f with output Hessians `hessians`
(one n×n matrix per output) for an input distributed N(x̄, P): the mean bias
bᵢ = ½·tr(Hᵢ·P) and the covariance correction Cᵢⱼ = ½·tr(Hᵢ·P·Hⱼ·P). Both are
exact for a quadratic f.
"""
function _second_order_terms(hessians, P::AbstractMatrix{T}) where {T}
    m = length(hessians)
    Pm = Matrix(P)
    HP = [hessians[i] * Pm for i in 1:m]
    bias = T[0.5 * tr(HP[i]) for i in 1:m]
    C = Matrix{T}(undef, m, m)
    for i in 1:m, j in i:m
        C[i, j] = C[j, i] = 0.5 * tr(HP[i] * HP[j])
    end
    return bias, C
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
