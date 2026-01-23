"""
Ensemble Kalman Filter (EnKF) - Monte Carlo approximation of the Kalman filter
for high-dimensional systems.

# Theory

The EnKF uses an ensemble of state vectors to represent the probability distribution:
```
{x₁⁽ⁱ⁾, ..., x_N⁽ⁱ⁾}  i = 1, ..., N_ens
```

Sample statistics replace the covariance matrix:
```
P̄ ≈ (1/(N-1)) ∑ᵢ (xⁱ - x̄)(xⁱ - x̄)ᵀ
```

The measurement update uses perturbed observations to maintain ensemble spread:
```
yⁱ = y + εⁱ,  εⁱ ~ N(0, R)
xⁱ⁺ = xⁱ⁻ + K(yⁱ - h(xⁱ⁻))
```

where the Kalman gain is computed from ensemble statistics.

# Features
- Handles very high-dimensional systems (atmosphere, ocean models)
- No need to store or propagate full covariance matrix
- Works with black-box nonlinear models
- Perturbed observations maintain ensemble spread
- Optional covariance localization for spurious correlations

# Complexity
- Time: O(N_ens · n) for ensemble propagation
- Space: O(N_ens · n) for ensemble storage
- Much more efficient than traditional KF when n >> N_ens

# Limitations
- Assumes Gaussian or near-Gaussian distributions
- Can suffer from filter divergence if N_ens too small
- Sampling error in covariance estimation
- May need localization for high-dimensional problems

# References
- 04-linear_filters.pdf
- Lecture24-EnKF.pdf
- Evensen (2003), "The Ensemble Kalman Filter: theoretical formulation and practical implementation"
- Burgers et al. (1998), "Analysis scheme in the ensemble Kalman filter"
"""
mutable struct EnsembleKalmanFilter{T<:Real, N} <: AbstractFilter
    ensemble::Matrix{T}                              # N × N_ens matrix
    ensemble_size::Int                               # Number of ensemble members
    
    # Inflation parameters (combat filter divergence)
    inflation_factor::T                              # Multiplicative inflation (≥ 1.0)
    additive_inflation::Union{Nothing, Matrix{T}}    # Additive inflation covariance
    
    # Localization (optional)
    use_localization::Bool
    localization_radius::Union{Nothing, T}
    localization_function::Union{Nothing, Function}
    
    # Diagnostics
    last_innovation::Union{Nothing, Vector{T}}
    last_innovation_cov::Union{Nothing, Matrix{T}}
    
    function EnsembleKalmanFilter{T, N}(
        ensemble::Matrix{T};
        inflation_factor::T = T(1.0),
        additive_inflation::Union{Nothing, Matrix{T}} = nothing,
        use_localization::Bool = false,
        localization_radius::Union{Nothing, T} = nothing,
        localization_function::Union{Nothing, Function} = nothing
    ) where {T<:Real, N}
        
        n_state, n_ens = size(ensemble)
        @assert n_state == N "Ensemble dimension must match filter dimension"
        @assert n_ens >= 2 "Ensemble size must be at least 2"
        @assert inflation_factor >= 1 "Inflation factor must be >= 1.0"
        
        new{T, N}(
            ensemble,
            n_ens,
            inflation_factor,
            additive_inflation,
            use_localization,
            localization_radius,
            localization_function,
            nothing,
            nothing
        )
    end
end

"""
    EnsembleKalmanFilter(x0, P0, n_ensemble; kwargs...)

Construct an EnKF by sampling from initial Gaussian distribution.

# Arguments
- `x0`: Initial state mean (SVector{N} or Vector)
- `P0`: Initial state covariance (SMatrix{N,N} or Matrix)
- `n_ensemble`: Number of ensemble members

# Keyword Arguments
- `inflation_factor`: Multiplicative covariance inflation (default: 1.0)
- `additive_inflation`: Additive inflation covariance matrix
- `use_localization`: Enable covariance localization (default: false)
- `localization_radius`: Localization radius in state-space distance
- `localization_function`: Custom localization function

# Example
```julia
x0 = @SVector zeros(4)
P0 = SMatrix{4,4}(1.0I)
enkf = EnsembleKalmanFilter(x0, P0, 50)
```
"""
function EnsembleKalmanFilter(
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
    
    return EnsembleKalmanFilter{T, N}(ensemble; kwargs...)
end

# ============================================================================
# Filter Operations
# ============================================================================

"""
    predict!(enkf::EnsembleKalmanFilter, model::AbstractDynamicsModel, u, dt)

Propagate each ensemble member through the dynamics model.

For each ensemble member i:
```
xⁱ⁺ = f(xⁱ, u, dt) + vⁱ
```

where vⁱ ~ N(0, Q) is sampled process noise.
"""
function predict!(
    enkf::EnsembleKalmanFilter{T, N},
    model::AbstractDynamicsModel,
    u,
    dt::Real
) where {T<:Real, N}
    
    # Propagate each ensemble member
    for i in 1:enkf.ensemble_size
        xⁱ = SVector{N, T}(enkf.ensemble[:, i])
        
        # Deterministic propagation
        xⁱ_new = propagate(model, xⁱ, u, dt)
        
        # Add process noise
        v = sample_process_noise(model, dt)
        xⁱ_new += v
        
        enkf.ensemble[:, i] = xⁱ_new
    end
    
    # Apply inflation
    apply_inflation!(enkf)
    
    return enkf
end

"""
    update!(enkf::EnsembleKalmanFilter, model::AbstractMeasurementModel, y)

Update ensemble with measurement using perturbed observations.

The EnKF update is:
```
xⁱ⁺ = xⁱ⁻ + K(yⁱ - h(xⁱ⁻))
```

where:
- yⁱ = y + εⁱ are perturbed observations, εⁱ ~ N(0, R)
- K = PₑHᵀ(HPₑHᵀ + R)⁻¹ is the ensemble Kalman gain
- Pₑ is the ensemble covariance
"""
function update!(
    enkf::EnsembleKalmanFilter{T, N},
    model::AbstractMeasurementModel,
    y::AbstractVector
) where {T<:Real, N}
    
    n_y = length(y)
    n_ens = enkf.ensemble_size
    
    # Compute ensemble mean (prior)
    x̄⁻ = vec(Statistics.mean(enkf.ensemble, dims=2))
    
    # Predicted measurements for each ensemble member
    H_ensemble = Matrix{T}(undef, n_y, n_ens)
    for i in 1:n_ens
        xⁱ = SVector{N, T}(enkf.ensemble[:, i])
        yⁱ = measure(model, xⁱ)
        H_ensemble[:, i] = yⁱ
    end
    
    # Ensemble mean of measurements
    ȳ⁻ = vec(Statistics.mean(H_ensemble, dims=2))
    
    # Innovation
    ν = y - ȳ⁻
    
    # Ensemble anomalies
    X_anom = enkf.ensemble .- x̄⁻
    Y_anom = H_ensemble .- ȳ⁻
    
    # Ensemble covariances (with Bessel correction)
    Pₑ_xy = (X_anom * Y_anom') / (n_ens - 1)
    Pₑ_yy = (Y_anom * Y_anom') / (n_ens - 1)
    
    # Measurement noise covariance
    R = measurement_noise_cov(model)
    
    # Innovation covariance
    S = Pₑ_yy + R
    S = make_symmetric(S)
    
    # Ensemble Kalman gain
    K = Pₑ_xy / S
    
    # Perturb observations
    R_sqrt = Matrix(cholesky(Symmetric(R)).L)
    perturbed_obs = Matrix{T}(undef, n_y, n_ens)
    for i in 1:n_ens
        ε = R_sqrt * randn(T, n_y)
        perturbed_obs[:, i] = y + ε
    end
    
    # Update each ensemble member
    for i in 1:n_ens
        yⁱ = perturbed_obs[:, i]
        ŷⁱ = H_ensemble[:, i]
        νⁱ = yⁱ - ŷⁱ
        
        enkf.ensemble[:, i] += K * νⁱ
    end
    
    # Store diagnostics
    enkf.last_innovation = ν
    enkf.last_innovation_cov = S
    
    return enkf
end

"""
    step!(enkf::EnsembleKalmanFilter, dyn, meas, u, y, dt)

Combined predict and update step.
"""
function step!(
    enkf::EnsembleKalmanFilter,
    dyn::AbstractDynamicsModel,
    meas::AbstractMeasurementModel,
    u,
    y::AbstractVector,
    dt::Real
)
    predict!(enkf, dyn, u, dt)
    update!(enkf, meas, y)
    return enkf
end

# ============================================================================
# State Estimation
# ============================================================================

"""
    state_estimate(enkf::EnsembleKalmanFilter)

Return the ensemble mean as the state estimate.
"""
function state_estimate(enkf::EnsembleKalmanFilter{T, N}) where {T<:Real, N}
    x̄ = vec(Statistics.mean(enkf.ensemble, dims=2))
    return SVector{N, T}(x̄)
end

"""
    covariance(enkf::EnsembleKalmanFilter)

Compute the ensemble covariance matrix.

```
P = (1/(N-1)) ∑ᵢ (xⁱ - x̄)(xⁱ - x̄)ᵀ
```
"""
function covariance(enkf::EnsembleKalmanFilter{T, N}) where {T<:Real, N}
    x̄ = vec(Statistics.mean(enkf.ensemble, dims=2))
    X_anom = enkf.ensemble .- x̄
    
    P = (X_anom * X_anom') / (enkf.ensemble_size - 1)
    return SMatrix{N, N, T, N*N}(P)
end

# ============================================================================
# Inflation
# ============================================================================

"""
    apply_inflation!(enkf::EnsembleKalmanFilter)

Apply covariance inflation to combat filter divergence.

Multiplicative inflation:
```
xⁱ = x̄ + α(xⁱ - x̄),  α ≥ 1
```

Additive inflation:
```
xⁱ = xⁱ + ηⁱ,  ηⁱ ~ N(0, Q_add)
```
"""
function apply_inflation!(enkf::EnsembleKalmanFilter{T, N}) where {T<:Real, N}
    # Multiplicative inflation
    if enkf.inflation_factor > 1.0
        x̄ = vec(Statistics.mean(enkf.ensemble, dims=2))
        for i in 1:enkf.ensemble_size
            enkf.ensemble[:, i] = x̄ + enkf.inflation_factor * (enkf.ensemble[:, i] - x̄)
        end
    end
    
    # Additive inflation
    if enkf.additive_inflation !== nothing
        Q_add = enkf.additive_inflation
        Q_sqrt = Matrix(cholesky(Symmetric(Q_add)).L)
        
        for i in 1:enkf.ensemble_size
            η = Q_sqrt * randn(T, N)
            enkf.ensemble[:, i] += η
        end
    end
    
    return enkf
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
    ensemble_spread(enkf::EnsembleKalmanFilter)

Compute the RMS ensemble spread (average standard deviation).
"""
function ensemble_spread(enkf::EnsembleKalmanFilter{T, N}) where {T<:Real, N}
    P = covariance(enkf)
    return sqrt(tr(P) / N)
end

"""
    innovation(enkf::EnsembleKalmanFilter)

Return the last innovation vector.
"""
innovation(enkf::EnsembleKalmanFilter) = enkf.last_innovation

"""
    innovation_covariance(enkf::EnsembleKalmanFilter)

Return the last innovation covariance.
"""
innovation_covariance(enkf::EnsembleKalmanFilter) = enkf.last_innovation_cov

"""
    nis(enkf::EnsembleKalmanFilter)

Compute Normalized Innovation Squared.
"""
function nis(enkf::EnsembleKalmanFilter{T}) where T
    if enkf.last_innovation === nothing || enkf.last_innovation_cov === nothing
        return T(NaN)
    end
    ν = enkf.last_innovation
    S = enkf.last_innovation_cov
    return ν' * (S \ ν)
end

"""
    num_ensemble(enkf::EnsembleKalmanFilter)

Return the ensemble size.
"""
num_ensemble(enkf::EnsembleKalmanFilter) = enkf.ensemble_size

