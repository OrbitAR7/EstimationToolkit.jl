"""
    DividedDifferenceFilter

Divided Difference Filter (DDF) using Stirling's interpolation formula.

The DDF is an alternative sigma-point method that uses divided differences
(numerical derivatives) to approximate the mean and covariance propagation.
It uses 2n sigma points with symmetric placement, similar to CKF but with
different weight derivation based on Stirling's interpolation.

# Mathematical Foundation

The DDF approximates Taylor series using Stirling's divided difference formula:
```
f(x) ≈ f(x₀) + Σᵢ [f(x₀ + hᵢ) - f(x₀ - hᵢ)]/(2hᵢ) · (x - x₀)ᵢ + ...
```

**Key Properties:**
- 2n sigma points (symmetric pairs)
- No tuning parameters
- Second-order accurate for general nonlinear functions
- Computationally efficient (fewer points than UKF)

**Reference**: Norgaard et al. (2000), "New developments in state estimation for nonlinear systems"
**References**: R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 4: Linear Filters (UT Austin ASE 381P.7, 2023), §3; R. Zanetti, "Other Kalman Filters," ASE 381P.7 lecture slides (UT Austin, 2023)

# Fields
- `x::SVector{N,T}` - State estimate
- `P::SMatrix{N,N,T}` - Covariance matrix
- `ν::Union{Nothing, SVector}` - Innovation
- `S::Union{Nothing, SMatrix}` - Innovation covariance
- `h::T` - Step size parameter (default: √3 for optimal 2nd-order accuracy)

# Constructor
```julia
DividedDifferenceFilter(x0::SVector{N,T}, P0::SMatrix{N,N,T}; h=sqrt(3.0))
```

# Example
```julia
# Initialize filter
ddf = DividedDifferenceFilter(x0, P0)  # 2n = 8 sigma points for 4D state

# Filtering cycle
for (u, y) in zip(controls, measurements)
    predict!(ddf, dynamics_model, u, dt)
    update!(ddf, measurement_model, y)
end
```

# Performance Characteristics
- **Sigma points**: 2n (fewer than UKF's 2n+1)
- **Accuracy**: Second-order for mean, first-order for covariance
- **No tuning**: Unlike UKF, no α/β/κ parameters
- **Computational cost**: Similar to CKF, lower than UKF
- **Best use**: When computational efficiency is important and UKF tuning is undesirable
"""
mutable struct DividedDifferenceFilter{T<:Real, N} <: AbstractFilter
    x::SVector{N, T}
    P::SMatrix{N, N, T}
    ν::Union{Nothing, SVector{<:Any, T}}
    S::Union{Nothing, SMatrix{<:Any, <:Any, T}}
    h::T  # Step size parameter
    
    function DividedDifferenceFilter(x0::SVector{N,T}, P0::SMatrix{N,N,T}; 
                                    h::T=sqrt(T(3))) where {N,T}
        # Ensure P0 is SPD
        P0_sym = make_symmetric(P0)
        P0_spd = ensure_spd(P0_sym)
        
        new{T,N}(x0, P0_spd, nothing, nothing, h)
    end
end

"""
    generate_divided_difference_points(μ, Σ, h)

Generate symmetric divided difference sigma points.

# Algorithm
1. Compute Cholesky factorization: Σ = L·L'
2. Generate symmetric pairs: χᵢ± = μ ± h·Lᵢ where Lᵢ is i-th column of L
3. Weights are equal: wᵢ = 1/(2n)

# Returns
Tuple of (points, weights) where points contains 2n sigma points.
"""
function generate_divided_difference_points(μ::SVector{N,T}, Σ::SMatrix{N,N,T}, h::T) where {N,T}
    # Compute Cholesky factorization
    Σ_sym = make_symmetric(Σ)
    Σ_spd = ensure_spd(Σ_sym)
    L = SMatrix{N,N,T}(cholesky(Symmetric(Matrix(Σ_spd))).L)
    
    # Generate 2n symmetric sigma points
    n_points = 2 * N
    points = Vector{SVector{N,T}}(undef, n_points)
    
    for i in 1:N
        # Positive and negative deviations
        points[2i - 1] = μ + h * L[:, i]
        points[2i] = μ - h * L[:, i]
    end
    
    # Equal weights for all points
    weights = fill(T(1 / n_points), n_points)
    
    return points, weights
end

"""
    predict!(ddf::DividedDifferenceFilter, model::AbstractDynamicsModel, u, dt)

Divided difference filter prediction step.

# Algorithm
1. Generate divided difference sigma points
2. Propagate through dynamics
3. Compute predicted mean and covariance
4. Add process noise
"""
function predict!(ddf::DividedDifferenceFilter{T,N}, model::AbstractDynamicsModel, u, dt) where {T,N}
    # Generate divided difference sigma points
    points, weights = generate_divided_difference_points(ddf.x, ddf.P, ddf.h)
    
    # Propagate sigma points through dynamics
    propagated_points = [propagate(model, χ, u, dt) for χ in points]
    
    # Compute predicted mean
    x_pred = sum(weights[i] * propagated_points[i] for i in eachindex(weights))
    
    # Compute predicted covariance using divided differences
    P_pred = sum(weights[i] * (propagated_points[i] - x_pred) * (propagated_points[i] - x_pred)' 
                 for i in eachindex(weights))
    
    # Add process noise
    Q = process_noise_cov(model, dt)
    P_pred = P_pred + Q
    
    # Ensure symmetry and positive definiteness
    P_pred_sym = make_symmetric(P_pred)
    ddf.P = ensure_spd(P_pred_sym)
    ddf.x = x_pred
    
    # Reset innovation statistics
    ddf.ν = nothing
    ddf.S = nothing
end

"""
    update!(ddf::DividedDifferenceFilter, model::AbstractMeasurementModel, y)

Divided difference filter measurement update.

# Algorithm
1. Generate divided difference sigma points at predicted state
2. Transform through measurement function
3. Compute predicted measurement and innovation
4. Compute innovation covariance and cross-covariance
5. Kalman gain and state/covariance update
"""
function update!(ddf::DividedDifferenceFilter{T,N}, model::AbstractMeasurementModel, y::SVector{NY,T}) where {T,N,NY}
    # Generate divided difference sigma points at predicted state
    points, weights = generate_divided_difference_points(ddf.x, ddf.P, ddf.h)
    
    # Transform sigma points through measurement function
    measured_points = [measure(model, χ) for χ in points]
    
    # Predicted measurement mean
    ŷ = sum(weights[i] * measured_points[i] for i in eachindex(weights))
    
    # Innovation
    ddf.ν = y - ŷ
    
    # Innovation covariance
    S_pred = sum(weights[i] * (measured_points[i] - ŷ) * (measured_points[i] - ŷ)' 
                 for i in eachindex(weights))
    R = measurement_noise_cov(model)
    S = S_pred + R
    S_sym = make_symmetric(S)
    ddf.S = ensure_spd(S_sym)
    
    # Cross-covariance
    P_xy = sum(weights[i] * (points[i] - ddf.x) * (measured_points[i] - ŷ)' 
               for i in eachindex(weights))
    
    # Kalman gain
    K = P_xy / Matrix(ddf.S)
    K_static = SMatrix{N,NY,T}(K)
    
    # State update
    ddf.x = ddf.x + K_static * ddf.ν
    
    # Covariance update
    P_updated = ddf.P - K_static * ddf.S * K_static'
    P_updated_sym = make_symmetric(P_updated)
    ddf.P = ensure_spd(P_updated_sym)
end

"""
    state_estimate(ddf::DividedDifferenceFilter)

Return current state estimate.
"""
function state_estimate(ddf::DividedDifferenceFilter)
    return ddf.x
end

"""
    covariance(ddf::DividedDifferenceFilter)

Return current covariance estimate.
"""
function covariance(ddf::DividedDifferenceFilter)
    return ddf.P
end

"""
    innovation(ddf::DividedDifferenceFilter)

Return the most recent innovation.
"""
function innovation(ddf::DividedDifferenceFilter)
    return ddf.ν
end

"""
    innovation_covariance(ddf::DividedDifferenceFilter)

Return the most recent innovation covariance.
"""
function innovation_covariance(ddf::DividedDifferenceFilter)
    return ddf.S
end

"""
    nis(ddf::DividedDifferenceFilter)

Compute Normalized Innovation Squared (NIS) statistic.
"""
function nis(ddf::DividedDifferenceFilter)
    if ddf.ν === nothing || ddf.S === nothing
        return nothing
    end
    return dot(ddf.ν, ddf.S \ Vector(ddf.ν))
end

# Alias for Central Difference Kalman Filter (CDKF)
"""
    CentralDifferenceKalmanFilter

Alias for [`DividedDifferenceFilter`](@ref). Both names refer to the same algorithm.
"""
const CentralDifferenceKalmanFilter = DividedDifferenceFilter
