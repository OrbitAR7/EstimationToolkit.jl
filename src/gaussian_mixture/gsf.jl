"""
    GaussianSumFilter

Gaussian Sum Filter (GSF) - Bank of Extended Kalman Filters with weighted components
for multi-modal posterior distributions.

# Theory

The GSF approximates the posterior as a Gaussian mixture:
```
p(x_k | y_{1:k}) ≈ ∑ᵢ wᵢ N(x_k; μᵢ, Pᵢ)
```

Each component is propagated using an EKF, and weights are updated based on
measurement likelihood:
```
wᵢ⁽ᵏ⁾ = wᵢ⁽ᵏ⁻¹⁾ · p(y_k | component i) / normalization
```

# Features
- Maintains multiple hypotheses for multi-modal distributions
- Each component is a standard EKF
- Automatic weight normalization
- Component pruning for efficiency (low-weight removal)
- Component merging to limit mixture size

# Complexity
- Time: O(M · n²) where M is number of components, n is state dimension
- Space: O(M · n²) for M covariance matrices

# Limitations
- Number of components can grow exponentially without pruning
- Requires tuning of pruning/merging thresholds
- Assumes Gaussian mixture is adequate representation

# References
- EstimationNotes Ch. 14
- 03-Nonlinear_filters.pdf §7
- Lecture15-GSF.pdf
- Alspach & Sorenson (1972), "Nonlinear Bayesian estimation using Gaussian sum approximations"
"""
mutable struct GaussianSumFilter{T<:Real, N} <: AbstractFilter
    components::Vector{ExtendedKalmanFilter{T, N}}  # Bank of EKFs
    weights::Vector{T}                              # Component weights (sum to 1)
    
    # Pruning parameters
    weight_threshold::T                             # Minimum weight to keep component
    max_components::Int                             # Maximum number of components
    
    # Diagnostics
    last_innovation::Union{Nothing, Vector{T}}
    last_innovation_cov::Union{Nothing, Matrix{T}}
    
    function GaussianSumFilter{T, N}(
        components::Vector{ExtendedKalmanFilter{T, N}},
        weights::Vector{T};
        weight_threshold::T = T(1e-4),
        max_components::Int = 20
    ) where {T<:Real, N}
        M = length(components)
        @assert M == length(weights) "Number of components must match weights"
        @assert M > 0 "Must have at least one component"
        @assert abs(sum(weights) - 1) < 1e-6 "Weights must sum to 1"
        @assert all(weights .>= 0) "All weights must be non-negative"
        @assert weight_threshold > 0 "Weight threshold must be positive"
        @assert max_components > 0 "Max components must be positive"
        
        new{T, N}(
            components,
            weights,
            weight_threshold,
            max_components,
            nothing,
            nothing
        )
    end
end

"""
    GaussianSumFilter(components, weights; kwargs...)

Construct a Gaussian Sum Filter from a vector of EKF components and weights.

# Arguments
- `components`: Vector of ExtendedKalmanFilter instances
- `weights`: Vector of component weights (must sum to 1)

# Keyword Arguments
- `weight_threshold`: Minimum weight to keep component (default: 1e-4)
- `max_components`: Maximum number of components (default: 20)

# Example
```julia
# Create initial components
ekf1 = ExtendedKalmanFilter(x1, P1)
ekf2 = ExtendedKalmanFilter(x2, P2)

# Create GSF with equal weights
gsf = GaussianSumFilter([ekf1, ekf2], [0.5, 0.5])
```
"""
function GaussianSumFilter(
    components::Vector{ExtendedKalmanFilter{T, N}},
    weights::Vector{T};
    kwargs...
) where {T<:Real, N}
    return GaussianSumFilter{T, N}(components, weights; kwargs...)
end

"""
    GaussianSumFilter(gmm::GMMBelief; kwargs...)

Construct a Gaussian Sum Filter from a Gaussian mixture belief.
"""
function GaussianSumFilter(
    gmm::GMMBelief{T, N};
    kwargs...
) where {T<:Real, N}
    components = [ExtendedKalmanFilter(comp.μ, comp.Σ) for comp in gmm.components]
    return GaussianSumFilter(components, gmm.weights; kwargs...)
end

# ============================================================================
# Filter Operations
# ============================================================================

"""
    predict!(gsf::GaussianSumFilter, model::AbstractDynamicsModel, u, dt)

Propagate all components through the dynamics model.

Each EKF component is predicted independently using the same dynamics model.
Weights remain unchanged during prediction.
"""
function predict!(
    gsf::GaussianSumFilter{T, N},
    model::AbstractDynamicsModel,
    u,
    dt::Real
) where {T<:Real, N}
    # Predict each component independently
    for component in gsf.components
        predict!(component, model, u, dt)
    end
    
    return gsf
end

"""
    update!(gsf::GaussianSumFilter, model::AbstractMeasurementModel, y)

Update all components with measurement and reweight based on likelihood.

The weight update is:
```
wᵢ⁽ᵏ⁾ = wᵢ⁽ᵏ⁻¹⁾ · 𝒩(y; ŷᵢ, Sᵢ)
```

where ŷᵢ is the predicted measurement and Sᵢ is the innovation covariance
for component i.
"""
function update!(
    gsf::GaussianSumFilter{T, N},
    model::AbstractMeasurementModel,
    y::AbstractVector
) where {T<:Real, N}
    M = length(gsf.components)
    likelihoods = zeros(T, M)
    
    # Update each component and compute likelihood
    for (i, component) in enumerate(gsf.components)
        # Perform EKF update
        update!(component, model, y)
        
        # Compute likelihood: p(y | component i)
        ν = component.last_innovation
        S = component.last_innovation_cov
        
        # Gaussian likelihood: 𝒩(y; ŷ, S) = exp(-0.5 * ν'S⁻¹ν) / √((2π)ⁿ|S|)
        n_y = length(y)
        
        # Mahalanobis distance
        d = ν' * (S \ ν)
        
        # Log-likelihood for numerical stability
        log_likelihood = -0.5 * (d + n_y * log(2π) + log(det(S)))
        likelihoods[i] = exp(log_likelihood)
    end
    
    # Update weights: w_new = w_old * likelihood
    new_weights = gsf.weights .* likelihoods
    
    # Normalize
    weight_sum = sum(new_weights)
    if weight_sum > eps(T)
        gsf.weights .= new_weights ./ weight_sum
    else
        # If all likelihoods are near zero, keep old weights
        @warn "All component likelihoods near zero, keeping old weights"
    end
    
    # Store mixture-level diagnostics (weighted average)
    gsf.last_innovation = sum(w * comp.last_innovation for (w, comp) in zip(gsf.weights, gsf.components))
    gsf.last_innovation_cov = sum(w * comp.last_innovation_cov for (w, comp) in zip(gsf.weights, gsf.components))
    
    # Prune low-weight components
    prune_components!(gsf)
    
    return gsf
end

"""
    step!(gsf::GaussianSumFilter, dyn, meas, u, y, dt)

Combined predict and update step.
"""
function step!(
    gsf::GaussianSumFilter,
    dyn::AbstractDynamicsModel,
    meas::AbstractMeasurementModel,
    u,
    y::AbstractVector,
    dt::Real
)
    predict!(gsf, dyn, u, dt)
    update!(gsf, meas, y)
    return gsf
end

# ============================================================================
# State Estimation
# ============================================================================

"""
    state_estimate(gsf::GaussianSumFilter)

Compute the weighted mean of all components.

```
x̂ = ∑ᵢ wᵢ μᵢ
```
"""
function state_estimate(gsf::GaussianSumFilter{T, N}) where {T<:Real, N}
    x̂ = zeros(SVector{N, T})
    for (w, component) in zip(gsf.weights, gsf.components)
        x̂ += w * state_estimate(component)
    end
    return x̂
end

"""
    covariance(gsf::GaussianSumFilter)

Compute the weighted covariance of all components using the mixture formula:

```
P = ∑ᵢ wᵢ [Pᵢ + (μᵢ - μ̄)(μᵢ - μ̄)ᵀ]
```

where μ̄ is the weighted mean.
"""
function covariance(gsf::GaussianSumFilter{T, N}) where {T<:Real, N}
    x̄ = state_estimate(gsf)
    P = zeros(SMatrix{N, N, T, N*N})
    
    for (w, component) in zip(gsf.weights, gsf.components)
        xᵢ = state_estimate(component)
        Pᵢ = covariance(component)
        
        # Within-component covariance
        P += w * Pᵢ
        
        # Between-component covariance (spread)
        Δx = xᵢ - x̄
        P += w * (Δx * Δx')
    end
    
    return P
end

# ============================================================================
# Component Management
# ============================================================================

"""
    prune_components!(gsf::GaussianSumFilter)

Remove components with weights below threshold and limit total number.

Components are pruned if:
1. Weight < weight_threshold
2. After sorting by weight, keep only max_components largest

Remaining weights are renormalized to sum to 1.
"""
function prune_components!(gsf::GaussianSumFilter{T, N}) where {T<:Real, N}
    # Find components above weight threshold
    keep_idx = findall(w -> w >= gsf.weight_threshold, gsf.weights)
    
    if length(keep_idx) == 0
        # Keep at least one component (the highest weight)
        keep_idx = [argmax(gsf.weights)]
    end
    
    # Limit to max_components (keep highest weight)
    if length(keep_idx) > gsf.max_components
        # Sort by weight (descending)
        sorted_idx = keep_idx[sortperm(gsf.weights[keep_idx], rev=true)]
        keep_idx = sorted_idx[1:gsf.max_components]
    end
    
    # Update components and weights
    if length(keep_idx) < length(gsf.components)
        gsf.components = gsf.components[keep_idx]
        gsf.weights = gsf.weights[keep_idx]
        
        # Renormalize
        gsf.weights ./= sum(gsf.weights)
    end
    
    return gsf
end

"""
    num_components(gsf::GaussianSumFilter)

Return the current number of components in the mixture.
"""
num_components(gsf::GaussianSumFilter) = length(gsf.components)

"""
    effective_components(gsf::GaussianSumFilter)

Compute the effective number of components using Shannon entropy:

```
N_eff = exp(-∑ᵢ wᵢ log wᵢ)
```

This measures the diversity of the mixture. A value close to M means
all components have similar weights, while a value near 1 means one
component dominates.
"""
function effective_components(gsf::GaussianSumFilter{T}) where T
    entropy = -sum(w * log(max(w, eps(T))) for w in gsf.weights)
    return exp(entropy)
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
    innovation(gsf::GaussianSumFilter)

Return the weighted average innovation across all components.
"""
function innovation(gsf::GaussianSumFilter)
    return gsf.last_innovation
end

"""
    innovation_covariance(gsf::GaussianSumFilter)

Return the weighted average innovation covariance.
"""
function innovation_covariance(gsf::GaussianSumFilter)
    return gsf.last_innovation_cov
end

"""
    nis(gsf::GaussianSumFilter)

Compute Normalized Innovation Squared for mixture.
"""
function nis(gsf::GaussianSumFilter{T}) where T
    if gsf.last_innovation === nothing || gsf.last_innovation_cov === nothing
        return T(NaN)
    end
    
    ν = gsf.last_innovation
    S = gsf.last_innovation_cov
    return ν' * (S \ ν)
end
