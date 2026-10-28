"""
    StaticMultipleModelFilter{T<:Real, N, M}

Static Multiple Model (SMM) filter for systems with unknown parameters.

The SMM filter runs a bank of filters in parallel, each assuming a different
value for an unknown constant parameter (e.g., target acceleration level,
ballistic coefficient, etc.). The overall estimate is a weighted combination
of all model-conditioned estimates.

Unlike the IMM (Interacting Multiple Model), the SMM assumes the parameter
does not switch over time - it's static but unknown.

# Mathematical Formulation

Models: θ ∈ {θ₁, θ₂, ..., θₘ}

Model probability:
```math
μⱼ(k) = P(θ = θⱼ | y₁:ₖ)
```

Model-conditioned filtering (run filter j for model θⱼ):
```math
x̂ⱼ(k|k), Pⱼ(k|k)
```

Model probability update (Bayes' rule):
```math
μⱼ(k) ∝ Λⱼ(k) μⱼ(k-1)
```

where Λⱼ(k) is the likelihood of measurement yₖ under model j.

Overall estimate (MMSE):
```math
x̂(k) = ∑ⱼ μⱼ(k) x̂ⱼ(k)
P(k) = ∑ⱼ μⱼ(k)[Pⱼ(k) + (x̂ⱼ(k) - x̂(k))(x̂ⱼ(k) - x̂(k))']
```

# Reference
- EstimationNotes.pdf Ch. 14 (Multiple Model Estimation)
- Bar-Shalom et al. (2001) - Estimation with Applications to Tracking

# Fields
- `filters::Vector{F}` - Bank of filters (one per model)
- `model_probs::Vector{T}` - Model probabilities μⱼ
- `likelihoods::Vector{T}` - Log measurement likelihoods per model (last update)
"""
mutable struct StaticMultipleModelFilter{T<:Real, N, F<:AbstractFilter} <: AbstractFilter
    filters::Vector{F}
    model_probs::Vector{T}
    likelihoods::Vector{T}
    
    # Overall estimate
    x̂::SVector{N, T}
    P::SMatrix{N, N, T}
end

"""
    StaticMultipleModelFilter(filters, initial_probs)

Create a Static Multiple Model filter from a bank of filters.

# Arguments
- `filters::Vector{F}` - Vector of initialized filters (one per model)
- `initial_probs::Vector{T}` - Initial model probabilities (must sum to 1)
"""
function StaticMultipleModelFilter(
    filters::Vector{F},
    initial_probs::Vector{T} = fill(T(1.0) / length(filters), length(filters))
) where {T<:Real, F<:AbstractFilter}
    
    M = length(filters)
    @assert M > 0 "Need at least one filter"
    @assert length(initial_probs) == M "Initial probabilities must match number of filters"
    @assert abs(sum(initial_probs) - 1.0) < 1e-6 "Probabilities must sum to 1"
    @assert all(initial_probs .>= 0) "Probabilities must be non-negative"
    
    likelihoods = zeros(T, M)
    
    # Compute initial overall estimate
    x̂ = compute_overall_estimate(filters, initial_probs)
    P = compute_overall_covariance(filters, initial_probs, x̂)
    N = length(x̂)
    
    return StaticMultipleModelFilter{T, N, F}(
        filters, initial_probs, likelihoods, x̂, P
    )
end

"""
    predict!(smm::StaticMultipleModelFilter, models_dyn, u, dt)

Prediction step: propagate each filter independently.

# Arguments
- `models_dyn::Vector` - Dynamics models (one per filter)
- `u` - Control input
- `dt` - Time step
"""
function predict!(
    smm::StaticMultipleModelFilter{T, N, F},
    models_dyn::Vector,
    u::AbstractVector = zeros(T, 0),
    dt::T = T(1.0)
) where {T, N, F}
    
    M = length(smm.filters)
    @assert length(models_dyn) == M "Need one dynamics model per filter"
    
    # Propagate each filter with its corresponding model
    for j in 1:M
        predict!(smm.filters[j], models_dyn[j], u, dt)
    end
    
    # Model probabilities remain constant in prediction (static parameter)
    # μⱼ(k+1|k) = μⱼ(k|k)
    
    return smm
end

"""
    update!(smm::StaticMultipleModelFilter, models_meas, y)

Measurement update: update each filter and recompute model probabilities.

# Arguments
- `models_meas::Vector` - Measurement models (one per filter)
- `y` - Measurement
"""
function update!(
    smm::StaticMultipleModelFilter{T, N, F},
    models_meas::Vector,
    y::AbstractVector
) where {T, N, F}
    
    M = length(smm.filters)
    @assert length(models_meas) == M "Need one measurement model per filter"
    
    # Update each filter and compute likelihood
    for j in 1:M
        # Compute innovation and its covariance before update
        ν_j, S_j = get_predicted_innovation_stats(smm.filters[j], models_meas[j], y)
        
        # Measurement log-likelihood: log Λⱼ = log N(y; ŷⱼ, Sⱼ)
        smm.likelihoods[j] = gaussian_loglikelihood(ν_j, S_j)
        
        # Perform measurement update
        update!(smm.filters[j], models_meas[j], y)
    end
    
    # Update model probabilities using Bayes' rule
    # μⱼ(k) ∝ Λⱼ(k) * μⱼ(k-1), in log space so large innovations don't underflow
    logw = smm.likelihoods .+ log.(max.(smm.model_probs, floatmin(T)))
    smm.model_probs .= normalize_log_weights(logw)
    
    # Compute overall MMSE estimate
    smm.x̂ = compute_overall_estimate(smm.filters, smm.model_probs)
    smm.P = compute_overall_covariance(smm.filters, smm.model_probs, smm.x̂)
    
    return smm
end

"""
    step!(smm, models_dyn, models_meas, u, y, dt)

Combined predict and update step.
"""
function step!(
    smm::StaticMultipleModelFilter{T, N, F},
    models_dyn::Vector,
    models_meas::Vector,
    u::AbstractVector,
    y::AbstractVector,
    dt::T = T(1.0)
) where {T, N, F}
    
    predict!(smm, models_dyn, u, dt)
    update!(smm, models_meas, y)
    return smm
end

"""
    compute_overall_estimate(filters, model_probs)

Compute weighted average of model-conditioned estimates.
"""
function compute_overall_estimate(filters::Vector{F}, model_probs::Vector{T}) where {F, T}
    N = length(state_estimate(filters[1]))
    x̂ = zeros(SVector{N, T})
    
    for (j, filt) in enumerate(filters)
        x̂ += model_probs[j] * state_estimate(filt)
    end
    
    return x̂
end

"""
    compute_overall_covariance(filters, model_probs, x̂_overall)

Compute overall covariance including spread of means.
"""
function compute_overall_covariance(
    filters::Vector{F},
    model_probs::Vector{T},
    x̂_overall::SVector{N, T}
) where {F, T, N}
    
    P = zeros(SMatrix{N, N, T})
    
    for (j, filt) in enumerate(filters)
        x̂_j = state_estimate(filt)
        P_j = covariance(filt)
        
        # Covariance: Pⱼ + spread term (x̂ⱼ - x̂)(x̂ⱼ - x̂)'
        diff = x̂_j - x̂_overall
        P += model_probs[j] * (P_j + diff * diff')
    end
    
    return make_symmetric(P)
end

"""
    gaussian_likelihood(innovation, S)

Compute Gaussian likelihood N(0; ν, S).
"""
function gaussian_likelihood(ν::AbstractVector{T}, S::AbstractMatrix{T}) where {T}
    n = length(ν)
    
    # Avoid numerical issues
    if !isposdef(S)
        S = make_symmetric(S)
        S = ensure_spd(S)
    end
    
    # Log-likelihood for numerical stability
    log_det = logdet(S)
    mahal = dot(ν, S \ ν)
    
    log_like = -0.5 * (n * log(2π) + log_det + mahal)
    
    return exp(log_like)
end

"""
    gaussian_loglikelihood(innovation, S)

Log of the Gaussian likelihood N(ν; 0, S). Mode probabilities are computed from
these rather than from `gaussian_likelihood`, because with tight measurements a
large innovation (a maneuver, a fault) makes exp() underflow to zero for every
model at once.
"""
function gaussian_loglikelihood(ν::AbstractVector{T}, S::AbstractMatrix{T}) where {T}
    n = length(ν)
    if !isposdef(S)
        S = ensure_spd(make_symmetric(S))
    end
    return -0.5 * (n * log(2π) + logdet(S) + dot(ν, S \ ν))
end

"""
    normalize_log_weights(logw)

Return exp.(logw) normalized to sum to one, using the max-shift trick.
"""
function normalize_log_weights(logw::AbstractArray{T}) where {T}
    m = maximum(logw)
    w = exp.(logw .- m)
    return w ./ sum(w)
end

"""
    state_estimate(smm::StaticMultipleModelFilter)

Return overall MMSE estimate.
"""
function state_estimate(smm::StaticMultipleModelFilter)
    return smm.x̂
end

"""
    covariance(smm::StaticMultipleModelFilter)

Return overall covariance.
"""
function covariance(smm::StaticMultipleModelFilter)
    return smm.P
end

"""
    get_model_probabilities(smm::StaticMultipleModelFilter)

Return current model probabilities.
"""
function get_model_probabilities(smm::StaticMultipleModelFilter)
    return smm.model_probs
end

"""
    get_model_estimates(smm::StaticMultipleModelFilter)

Return state estimates from each model-conditioned filter.
"""
function get_model_estimates(smm::StaticMultipleModelFilter)
    return [state_estimate(f) for f in smm.filters]
end

"""
    most_probable_model(smm::StaticMultipleModelFilter)

Return index of model with highest probability.
"""
function most_probable_model(smm::StaticMultipleModelFilter)
    return argmax(smm.model_probs)
end
