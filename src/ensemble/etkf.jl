"""
Ensemble Transform Kalman Filter (ETKF) - Deterministic ensemble update
without perturbed observations.

# Theory

The ETKF is a square-root ensemble filter that updates the ensemble
subspace directly using a transform matrix T:

```
Xᵃ = X̄ᶠ + Xᶠ T
```

where:
- Xᶠ is the forecast ensemble anomaly matrix
- T is the transform matrix from the Kalman update equations
- X̄ᶠ is updated using the ensemble mean Kalman update

The transform maintains the correct analysis covariance without
requiring perturbed observations.

# Advantages over Standard EnKF
- No perturbed observations (deterministic update)
- Better sampling properties with small ensembles
- Preserves ensemble subspace structure
- No extra sampling error from observation perturbations

# Features
- Deterministic analysis update
- Square-root formulation for stability
- Works with nonlinear measurement models
- Optional localization support

# Complexity
- Time: O(N_ens³ + N_ens² · n_y) for transform computation
- Space: O(N_ens · n) for ensemble storage

# Limitations
- Assumes Gaussian or near-Gaussian distributions
- More complex implementation than standard EnKF
- Transform computation can be expensive for large N_y

# References
- R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 4: Linear Filters (UT Austin ASE 381P.7, 2023), §4
- Bishop et al. (2001), "Adaptive sampling with the ensemble transform Kalman filter"
- Hunt et al. (2007), "Efficient data assimilation for spatiotemporal chaos"
"""
mutable struct EnsembleTransformKalmanFilter{T<:Real, N} <: AbstractFilter
    ensemble::Matrix{T}                              # N × N_ens matrix
    ensemble_size::Int                               # Number of ensemble members
    
    # Inflation parameters
    inflation_factor::T                              # Multiplicative inflation (≥ 1.0)
    
    # Localization (optional)
    use_localization::Bool
    localization_radius::Union{Nothing, T}
    
    # Diagnostics
    last_innovation::Union{Nothing, Vector{T}}
    last_innovation_cov::Union{Nothing, Matrix{T}}
    
    function EnsembleTransformKalmanFilter{T, N}(
        ensemble::Matrix{T};
        inflation_factor::T = T(1.0),
        use_localization::Bool = false,
        localization_radius::Union{Nothing, T} = nothing
    ) where {T<:Real, N}
        
        n_state, n_ens = size(ensemble)
        @assert n_state == N "Ensemble dimension must match filter dimension"
        @assert n_ens >= 2 "Ensemble size must be at least 2"
        @assert inflation_factor >= 1 "Inflation factor must be >= 1.0"
        
        new{T, N}(
            ensemble,
            n_ens,
            inflation_factor,
            use_localization,
            localization_radius,
            nothing,
            nothing
        )
    end
end

"""
    EnsembleTransformKalmanFilter(x0, P0, n_ensemble; kwargs...)

Construct an ETKF by sampling from initial Gaussian distribution.

# Arguments
- `x0`: Initial state mean
- `P0`: Initial state covariance
- `n_ensemble`: Number of ensemble members

# Keyword Arguments
- `inflation_factor`: Multiplicative covariance inflation (default: 1.0)
- `use_localization`: Enable covariance localization (default: false)
- `localization_radius`: Localization radius in state-space distance

# Example
```julia
x0 = @SVector zeros(4)
P0 = SMatrix{4,4}(1.0I)
etkf = EnsembleTransformKalmanFilter(x0, P0, 50)
```
"""
function EnsembleTransformKalmanFilter(
    x0::AbstractVector{T},
    P0::AbstractMatrix{T},
    n_ensemble::Int;
    kwargs...
) where T<:Real
    
    N = length(x0)
    @assert size(P0) == (N, N)
    @assert is_positive_definite(P0)
    
    # Sample ensemble from initial distribution
    dist = MvNormal(Vector(x0), Matrix(P0))
    ensemble = Matrix{T}(undef, N, n_ensemble)
    
    for i in 1:n_ensemble
        ensemble[:, i] = rand(dist)
    end
    
    return EnsembleTransformKalmanFilter{T, N}(ensemble; kwargs...)
end

# ============================================================================
# Filter Operations
# ============================================================================

"""
    predict!(etkf::EnsembleTransformKalmanFilter, model::AbstractDynamicsModel, u, dt)

Propagate each ensemble member through the dynamics model.
"""
function predict!(
    etkf::EnsembleTransformKalmanFilter{T, N},
    model::AbstractDynamicsModel,
    u,
    dt::Real
) where {T<:Real, N}
    
    # Propagate each ensemble member
    for i in 1:etkf.ensemble_size
        xⁱ = SVector{N, T}(etkf.ensemble[:, i])
        
        # Deterministic propagation
        xⁱ_new = propagate(model, xⁱ, u, dt)
        
        # Add process noise
        v = sample_process_noise(model, dt)
        xⁱ_new += v
        
        etkf.ensemble[:, i] = xⁱ_new
    end
    
    # Apply inflation
    apply_inflation!(etkf)
    
    return etkf
end

"""
    update!(etkf::EnsembleTransformKalmanFilter, model::AbstractMeasurementModel, y)

Update ensemble using the ensemble transform.

The ETKF update computes a transform matrix T such that:
```
Xᵃ = X̄ᶠ + Xᶠ T
```

where T satisfies:
```
(Nₑ - 1)T Tᵀ = [I + Yᶠᵀ R⁻¹ Yᶠ]⁻¹
```

and the ensemble mean is updated separately.
"""
function update!(
    etkf::EnsembleTransformKalmanFilter{T, N},
    model::AbstractMeasurementModel,
    y::AbstractVector
) where {T<:Real, N}
    
    n_y = length(y)
    n_ens = etkf.ensemble_size
    
    # Compute ensemble mean (prior)
    x̄⁻ = vec(Statistics.mean(etkf.ensemble, dims=2))
    
    # Predicted measurements for each ensemble member
    H_ensemble = Matrix{T}(undef, n_y, n_ens)
    for i in 1:n_ens
        xⁱ = SVector{N, T}(etkf.ensemble[:, i])
        yⁱ = measure(model, xⁱ)
        H_ensemble[:, i] = yⁱ
    end
    
    # Ensemble mean of measurements
    ȳ⁻ = vec(Statistics.mean(H_ensemble, dims=2))
    
    # Innovation
    ν = y - ȳ⁻
    
    # Ensemble anomalies (mean-removed)
    X_anom = etkf.ensemble .- x̄⁻
    Y_anom = H_ensemble .- ȳ⁻
    
    # Measurement noise covariance
    R = measurement_noise_cov(model)
    R_inv = inv(Symmetric(R))
    
    # Compute transform matrix
    # T satisfies: (N-1)T Tᵀ = [I + Yᵀ R⁻¹ Y]⁻¹
    
    # Normalize anomalies
    Y_normalized = Y_anom / sqrt(n_ens - 1)
    
    # Compute C = I + Yᵀ R⁻¹ Y
    C = I(n_ens) + Y_normalized' * R_inv * Y_normalized
    C = make_symmetric(C)
    
    # Compute C⁻¹ using eigenvalue decomposition for stability
    eigvals_C, eigvecs_C = eigen(Symmetric(C))
    C_inv = eigvecs_C * Diagonal(1 ./ eigvals_C) * eigvecs_C'
    
    # Transform matrix: Tmat = C⁻¹/² 
    # Compute square root of C⁻¹
    Tmat = eigvecs_C * Diagonal(1 ./ sqrt.(eigvals_C)) * eigvecs_C'
    
    # Update ensemble mean (standard Kalman update)
    Pₑ_xy = (X_anom * Y_anom') / (n_ens - 1)
    Pₑ_yy = (Y_anom * Y_anom') / (n_ens - 1)
    S = Pₑ_yy + R
    K = Pₑ_xy / S
    x̄ᵃ = x̄⁻ + K * ν
    
    # Update ensemble anomalies using transform
    X_anom_new = X_anom * Tmat
    
    # Reconstruct analysis ensemble
    etkf.ensemble = X_anom_new .+ x̄ᵃ
    
    # Store diagnostics
    etkf.last_innovation = ν
    etkf.last_innovation_cov = S
    
    return etkf
end

"""
    step!(etkf::EnsembleTransformKalmanFilter, dyn, meas, u, y, dt)

Combined predict and update step.
"""
function step!(
    etkf::EnsembleTransformKalmanFilter,
    dyn::AbstractDynamicsModel,
    meas::AbstractMeasurementModel,
    u,
    y::AbstractVector,
    dt::Real
)
    predict!(etkf, dyn, u, dt)
    update!(etkf, meas, y)
    return etkf
end

# ============================================================================
# State Estimation
# ============================================================================

"""
    state_estimate(etkf::EnsembleTransformKalmanFilter)

Return the ensemble mean as the state estimate.
"""
function state_estimate(etkf::EnsembleTransformKalmanFilter{T, N}) where {T<:Real, N}
    x̄ = vec(Statistics.mean(etkf.ensemble, dims=2))
    return SVector{N, T}(x̄)
end

"""
    covariance(etkf::EnsembleTransformKalmanFilter)

Compute the ensemble covariance matrix.
"""
function covariance(etkf::EnsembleTransformKalmanFilter{T, N}) where {T<:Real, N}
    x̄ = vec(Statistics.mean(etkf.ensemble, dims=2))
    X_anom = etkf.ensemble .- x̄
    
    P = (X_anom * X_anom') / (etkf.ensemble_size - 1)
    return SMatrix{N, N, T, N*N}(P)
end

# ============================================================================
# Inflation
# ============================================================================

"""
    apply_inflation!(etkf::EnsembleTransformKalmanFilter)

Apply multiplicative covariance inflation.
"""
function apply_inflation!(etkf::EnsembleTransformKalmanFilter{T, N}) where {T<:Real, N}
    if etkf.inflation_factor > 1.0
        x̄ = vec(Statistics.mean(etkf.ensemble, dims=2))
        for i in 1:etkf.ensemble_size
            etkf.ensemble[:, i] = x̄ + etkf.inflation_factor * (etkf.ensemble[:, i] - x̄)
        end
    end
    
    return etkf
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
    ensemble_spread(etkf::EnsembleTransformKalmanFilter)

Compute the RMS ensemble spread.
"""
function ensemble_spread(etkf::EnsembleTransformKalmanFilter{T, N}) where {T<:Real, N}
    P = covariance(etkf)
    return sqrt(tr(P) / N)
end

innovation(etkf::EnsembleTransformKalmanFilter) = etkf.last_innovation
innovation_covariance(etkf::EnsembleTransformKalmanFilter) = etkf.last_innovation_cov

function nis(etkf::EnsembleTransformKalmanFilter{T}) where T
    if etkf.last_innovation === nothing || etkf.last_innovation_cov === nothing
        return T(NaN)
    end
    ν = etkf.last_innovation
    S = etkf.last_innovation_cov
    return ν' * (S \ ν)
end

num_ensemble(etkf::EnsembleTransformKalmanFilter) = etkf.ensemble_size
