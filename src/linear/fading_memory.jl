"""
Fading Memory Filter - Kalman Filter with exponential forgetting factor.

The Fading Memory Filter (FMF) is a robust variant of the Kalman filter that
applies an exponential forgetting factor to the covariance matrix. This makes
the filter more responsive to model changes or sudden maneuvers by "forgetting"
older information.

# Mathematical Form

The FMF modifies the standard Kalman filter prediction step:

Standard KF: P̄(k+1) = F(k)P(k)Fᵀ(k) + Q(k)
Fading Memory: P̄(k+1) = α⁻²[F(k)P(k)Fᵀ(k) + Q(k)]

where α ∈ (0, 1] is the forgetting factor:
- α = 1.0: Standard Kalman filter (no forgetting)
- α < 1.0: Exponential forgetting (more responsive)
- Typical values: 0.95 - 0.99

# Use Cases
- Time-varying systems
- Maneuvering target tracking
- Model uncertainty
- Adaptive estimation

# Reference
- Jazwinski, A. H. (1970). "Stochastic Processes and Filtering Theory"
- Bar-Shalom et al. (2001). "Estimation with Applications to Tracking"
"""

"""
    FadingMemoryFilter{T<:Real, N}

Fading memory filter with exponential forgetting.

# Fields
- `x::SVector{N,T}`: State estimate
- `P::SMatrix{N,N,T}`: Covariance matrix
- `α::T`: Forgetting factor (0 < α ≤ 1)
- `ν::Union{Nothing, SVector}`: Innovation
- `S::Union{Nothing, SMatrix}`: Innovation covariance

# Constructor
```julia
FadingMemoryFilter(x0, P0; α=0.98)
```

# Example
```julia
using EstimationToolkit, StaticArrays

x0 = @SVector [0.0, 0.0, 1.0, 0.5]
P0 = @SMatrix [1.0 0.0 0.0 0.0;
               0.0 1.0 0.0 0.0;
               0.0 0.0 0.1 0.0;
               0.0 0.0 0.0 0.1]

# More responsive than standard KF
fmf = FadingMemoryFilter(x0, P0, α=0.95)

# Use like Kalman filter
predict!(fmf, dynamics, u, dt)
update!(fmf, measurement_model, z)
```
"""
mutable struct FadingMemoryFilter{T<:Real, N} <: AbstractFilter
    x::SVector{N, T}
    P::SMatrix{N, N, T}
    α::T  # Forgetting factor
    ν::Union{Nothing, SVector}
    S::Union{Nothing, SMatrix}
    
    function FadingMemoryFilter(x0::SVector{N, T}, P0::SMatrix{N, N, T}; 
                                α::T = T(0.98)) where {N, T<:Real}
        @assert 0 < α ≤ 1 "Forgetting factor α must be in (0, 1]"
        @assert is_positive_definite(P0) "Initial covariance must be positive definite"
        new{T, N}(x0, P0, α, nothing, nothing)
    end
end

"""
    predict!(fmf::FadingMemoryFilter, model_dyn, u, dt)

Prediction step with fading memory (inflated covariance).
"""
function predict!(
    fmf::FadingMemoryFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    u::AbstractVector = zeros(T, 0),
    dt::T = T(1.0)
) where {T, N}
    
    # Propagate state
    x_pred = propagate(model_dyn, fmf.x, u, dt)
    
    # Compute Jacobian
    F = jacobian_x(model_dyn, fmf.x, u, dt)
    F_static = SMatrix{N,N,T}(F)
    
    # Process noise
    Q = process_noise_cov(model_dyn, dt)
    Q_static = SMatrix{N,N,T}(Q)
    
    # Fading memory covariance propagation
    # P̄ = α⁻²[FPFᵀ + Q]
    α_inv_sq = 1 / (fmf.α^2)
    P_pred = α_inv_sq * (F_static * fmf.P * F_static' + Q_static)
    P_pred_sym = make_symmetric(P_pred)
    
    fmf.x = x_pred
    fmf.P = ensure_spd(P_pred_sym)
    
    return fmf
end

"""
    update!(fmf::FadingMemoryFilter, model_meas, y)

Measurement update (standard Kalman update).
"""
function update!(
    fmf::FadingMemoryFilter{T, N},
    model_meas::AbstractMeasurementModel,
    y::AbstractVector
) where {T, N}
    
    # Predicted measurement
    ŷ = measure(model_meas, fmf.x)
    
    # Innovation
    ν = SVector{length(y), T}(y - ŷ)
    fmf.ν = ν
    
    # Measurement Jacobian
    H = jacobian_x(model_meas, fmf.x)
    R = measurement_noise_cov(model_meas)
    
    NY = length(y)
    H_static = SMatrix{NY, N, T}(H)
    R_static = SMatrix{NY, NY, T}(R)
    
    # Innovation covariance
    S = H_static * fmf.P * H_static' + R_static
    S_sym = make_symmetric(S)
    S_spd = ensure_spd(S_sym)
    fmf.S = S_spd
    
    # Kalman gain
    K = fmf.P * H_static' / Matrix(S_spd)
    K_static = SMatrix{N, NY, T}(K)
    
    # State update
    fmf.x = fmf.x + K_static * ν
    
    # Covariance update (Joseph form)
    I_KH = SMatrix{N,N,T}(I) - K_static * H_static
    P_updated = I_KH * fmf.P * I_KH' + K_static * R_static * K_static'
    P_updated_sym = make_symmetric(P_updated)
    fmf.P = ensure_spd(P_updated_sym)
    
    return fmf
end

# Standard interface implementations
state_estimate(fmf::FadingMemoryFilter) = fmf.x
covariance(fmf::FadingMemoryFilter) = fmf.P
innovation(fmf::FadingMemoryFilter) = fmf.ν
innovation_covariance(fmf::FadingMemoryFilter) = fmf.S

function nis(fmf::FadingMemoryFilter)
    if fmf.ν === nothing || fmf.S === nothing
        return nothing
    end
    return dot(fmf.ν, fmf.S \ Vector(fmf.ν))
end

"""
    set_forgetting_factor!(fmf::FadingMemoryFilter, α)

Update the forgetting factor.

# Arguments
- `α::Real`: New forgetting factor (0 < α ≤ 1)
"""
function set_forgetting_factor!(fmf::FadingMemoryFilter{T}, α::Real) where {T}
    @assert 0 < α ≤ 1 "Forgetting factor α must be in (0, 1]"
    fmf.α = T(α)
    return fmf
end
