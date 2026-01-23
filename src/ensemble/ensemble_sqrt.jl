"""
    EnsembleSquareRootFilter{T<:Real, N}

Ensemble Square-Root Filter (EnSRF) - deterministic ensemble Kalman filter.

The EnSRF is a deterministic variant of the EnKF that avoids perturbing
observations. Instead, it uses a square-root update formula to directly
compute the analysis ensemble that yields the correct analysis covariance.

# Mathematical Formulation

Ensemble matrix: 𝐗 = [x₁, x₂, ..., xₙ] (N × Nₑₙₛ)

Sample mean and perturbations:
```math
x̄ = (1/Nₑₙₛ) ∑ᵢ xᵢ
𝐀 = 𝐗 - x̄𝟏ᵀ
```

Sample covariance:
```math
Pᵉⁿˢ = (1/(Nₑₙₛ-1)) 𝐀𝐀ᵀ
```

Square-root update (analysis perturbations):
```math
𝐀ᵃ = 𝐀ᶠ 𝐓
```

where 𝐓 is computed to satisfy:
```math
𝐀ᵃ(𝐀ᵃ)ᵀ = Pᵃ
```

# Advantages over EnKF
- Deterministic (no observation perturbations)
- Exactly preserves analysis covariance
- No spurious sampling error from perturbed observations
- Better for small ensembles

# Reference
- 04-linear_filters.pdf (Ensemble Methods)
- Whitaker & Hamill (2002)
- Tippett et al. (2003)
- Evensen (2009) - "Data Assimilation: The Ensemble Kalman Filter"

# Fields
- `ensemble::Matrix{T}` - Ensemble matrix (N × Nₑₙₛ)
- `x̄::SVector{N,T}` - Ensemble mean
- `N_ensemble::Int` - Number of ensemble members
"""
mutable struct EnsembleSquareRootFilter{T<:Real, N} <: AbstractFilter
    ensemble::Matrix{T}  # N × N_ensemble
    x̄::SVector{N, T}
    N_ensemble::Int
    
    # Localization (optional)
    localization::Union{Nothing, Function}
end

"""
    EnsembleSquareRootFilter(x0, P0, N_ensemble; localization=nothing)

Create Ensemble Square-Root Filter.

# Arguments
- `x0::SVector{N,T}` - Initial mean
- `P0::SMatrix{N,N,T}` - Initial covariance
- `N_ensemble::Int` - Number of ensemble members

# Keyword Arguments
- `localization::Union{Nothing, Function} = nothing` - Covariance localization function
"""
function EnsembleSquareRootFilter(
    x0::SVector{N, T},
    P0::SMatrix{N, N, T},
    N_ensemble::Int;
    localization::Union{Nothing, Function} = nothing
) where {T<:Real, N}
    
    @assert N_ensemble > 1 "Need at least 2 ensemble members"
    @assert is_positive_definite(P0) "Initial covariance must be SPD"
    
    # Generate initial ensemble by sampling
    L = cholesky(Symmetric(P0)).L
    ensemble = zeros(T, N, N_ensemble)
    
    for i in 1:N_ensemble
        z = randn(T, N)
        ensemble[:, i] = x0 + L * z
    end
    
    return EnsembleSquareRootFilter{T, N}(ensemble, x0, N_ensemble, localization)
end

"""
    predict!(ensrf::EnsembleSquareRootFilter, model_dyn, u, dt)

Prediction step: propagate each ensemble member.
"""
function predict!(
    ensrf::EnsembleSquareRootFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    u::AbstractVector = zeros(T, 0),
    dt::T = T(1.0)
) where {T, N}
    
    # Propagate each ensemble member
    for i in 1:ensrf.N_ensemble
        xᵢ = SVector{N, T}(view(ensrf.ensemble, :, i))
        
        # Deterministic propagation
        x_prop = propagate(model_dyn, xᵢ, u, dt)
        
        # Add process noise sample
        Q = process_noise_cov(model_dyn, dt)
        v = rand(MvNormal(zeros(N), Q))
        
        ensrf.ensemble[:, i] = x_prop + v
    end
    
    # Update mean
    ensrf.x̄ = SVector{N, T}(Statistics.mean(ensrf.ensemble, dims=2)[:])
    
    return ensrf
end

"""
    update!(ensrf::EnsembleSquareRootFilter, model_meas, y)

Square-root analysis update without observation perturbations.
"""
function update!(
    ensrf::EnsembleSquareRootFilter{T, N},
    model_meas::AbstractMeasurementModel,
    y::AbstractVector
) where {T, N}
    
    Nₑ = ensrf.N_ensemble
    n_y = length(y)
    
    # Ensemble mean
    x̄ᶠ = ensrf.x̄
    
    # Forecast perturbation matrix: 𝐀ᶠ = 𝐗 - x̄𝟏ᵀ
    Aᶠ = ensrf.ensemble .- x̄ᶠ  # N × Nₑ
    
    # Measurement operator applied to ensemble
    Y = zeros(T, n_y, Nₑ)
    for i in 1:Nₑ
        xᵢ = SVector{N, T}(view(ensrf.ensemble, :, i))
        Y[:, i] = measure(model_meas, xᵢ)
    end
    
    # Predicted measurement mean
    ȳᶠ = Statistics.mean(Y, dims=2)[:]
    
    # Measurement perturbations: HA = Y - ȳ𝟏ᵀ
    HA = Y .- ȳᶠ  # n_y × Nₑ
    
    # Innovation
    d = y - ȳᶠ
    
    # Measurement noise covariance
    R = measurement_noise_cov(model_meas)
    
    # Innovation covariance (sample-based)
    # S = (1/(Nₑ-1)) HAHAᵀ + R
    S = (HA * HA') / (Nₑ - 1) + R
    S = make_symmetric(S)
    
    # Kalman gain (ensemble-based)
    # K = (1/(Nₑ-1)) AᶠHAᵀ S⁻¹
    K = (Aᶠ * HA') / ((Nₑ - 1) * S)
    
    # Analysis mean
    x̄ᵃ = x̄ᶠ + K * d
    
    # Square-root update for perturbations
    # Compute transformation matrix T such that AᵃAᵃᵀ has correct covariance
    
    # Reduced-dimension update: HA ∈ ℝⁿʸˣᴺᵉ
    # U = (Nₑ-1) I + HAᵀ R⁻¹ HA
    U = (Nₑ - 1) * Matrix{T}(I, Nₑ, Nₑ) + HA' * (R \ HA)
    
    # Cholesky of U
    C = cholesky(Symmetric(U))
    U_sqrt_inv = inv(C.L)  # Lower triangular
    
    # Transformation matrix
    T_mat = sqrt(T(Nₑ - 1)) * U_sqrt_inv
    
    # Analysis perturbations
    Aᵃ = Aᶠ * T_mat
    
    # Analysis ensemble
    ensrf.ensemble = Aᵃ .+ x̄ᵃ
    ensrf.x̄ = x̄ᵃ
    
    return ensrf
end

"""
    step!(ensrf, model_dyn, model_meas, u, y, dt)

Combined predict and update step.
"""
function step!(
    ensrf::EnsembleSquareRootFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    model_meas::AbstractMeasurementModel,
    u::AbstractVector,
    y::AbstractVector,
    dt::T = T(1.0)
) where {T, N}
    
    predict!(ensrf, model_dyn, u, dt)
    update!(ensrf, model_meas, y)
    return ensrf
end

"""
    state_estimate(ensrf::EnsembleSquareRootFilter)

Return ensemble mean.
"""
function state_estimate(ensrf::EnsembleSquareRootFilter)
    return ensrf.x̄
end

"""
    covariance(ensrf::EnsembleSquareRootFilter)

Compute sample covariance from ensemble.
"""
function covariance(ensrf::EnsembleSquareRootFilter{T, N}) where {T, N}
    A = ensrf.ensemble .- ensrf.x̄
    P = (A * A') / (ensrf.N_ensemble - 1)
    return SMatrix{N, N, T}(make_symmetric(P))
end

"""
    num_ensemble(ensrf::EnsembleSquareRootFilter)

Return number of ensemble members.
"""
function num_ensemble(ensrf::EnsembleSquareRootFilter)
    return ensrf.N_ensemble
end

"""
    ensemble_spread(ensrf::EnsembleSquareRootFilter)

Compute RMS ensemble spread.
"""
function ensemble_spread(ensrf::EnsembleSquareRootFilter{T, N}) where {T, N}
    P = covariance(ensrf)
    return sqrt(tr(P) / N)
end

"""
    get_ensemble(ensrf::EnsembleSquareRootFilter)

Return ensemble matrix.
"""
function get_ensemble(ensrf::EnsembleSquareRootFilter)
    return ensrf.ensemble
end
