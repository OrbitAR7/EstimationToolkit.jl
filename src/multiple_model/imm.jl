"""
    InteractingMultipleModel{T<:Real, N, F}

Interacting Multiple Model (IMM) filter for systems with switching dynamics.

The IMM is a powerful algorithm for tracking systems that switch between
multiple dynamic models (e.g., constant velocity vs. maneuvering targets,
different flight modes). Unlike the Static MM filter, the IMM allows the
system mode to change over time according to a Markov chain.

# Mathematical Formulation

Models: j ∈ {1, 2, ..., M}

Markov transition matrix:
```math
πᵢⱼ = P(model j at k+1 | model i at k)
```

Mode probability:
```math
μⱼ(k) = P(model j at k | y₁:ₖ)
```

IMM Algorithm (four steps per cycle):

1. **Interaction (Mixing)**: Compute mixed initial conditions for each filter
   ```math
   x̂⁰ⱼ(k|k) = ∑ᵢ x̂ᵢ(k|k) μᵢ|ⱼ(k|k)
   ```
   where μᵢ|ⱼ = P(mode i at k | mode j at k+1, y₁:ₖ)

2. **Filtering**: Run each model-matched filter starting from mixed IC
   - Predict and update with model j

3. **Mode Probability Update**: Update mode probabilities
   ```math
   μⱼ(k+1) ∝ Λⱼ(k+1) c̄ⱼ
   ```
   where Λⱼ is measurement likelihood, c̄ⱼ is mode transition normalization

4. **Combination**: Compute overall estimate
   ```math
   x̂(k+1) = ∑ⱼ μⱼ(k+1) x̂ⱼ(k+1)
   ```

# Key Advantages
- Handles mode-switching dynamics (maneuvers, failures)
- Soft switching via probabilistic mode transitions
- Better than hard decisions or model selection

# Reference
- EstimationNotes.pdf §14.2.1 (IMM Algorithm)
- Blom & Bar-Shalom (1988)
- Bar-Shalom et al. (2001) - "Estimation with Applications to Tracking"

# Fields
- `filters::Vector{F}` - Bank of filters (one per model)
- `mode_probs::Vector{T}` - Current mode probabilities μⱼ(k)
- `transition_matrix::Matrix{T}` - Markov transition probabilities πᵢⱼ
- `mixing_probs::Matrix{T}` - Conditional probabilities μᵢ|ⱼ(k|k)
- `likelihoods::Vector{T}` - Measurement likelihoods per model
"""
mutable struct InteractingMultipleModel{T<:Real, N, F<:AbstractFilter} <: AbstractFilter
    filters::Vector{F}
    mode_probs::Vector{T}
    transition_matrix::Matrix{T}
    
    # Intermediate computations
    mixing_probs::Matrix{T}
    likelihoods::Vector{T}
    predicted_mode_probs::Vector{T}
    
    # Mixed initial conditions for each filter
    mixed_states::Vector{SVector{N, T}}
    mixed_covs::Vector{SMatrix{N, N, T}}
    
    # Overall estimate
    x̂::SVector{N, T}
    P::SMatrix{N, N, T}
end

"""
    InteractingMultipleModel(filters, transition_matrix, initial_probs)

Create an IMM filter.

# Arguments
- `filters::Vector{F}` - Vector of initialized filters
- `transition_matrix::Matrix{T}` - Mode transition probabilities πᵢⱼ (row stochastic)
- `initial_probs::Vector{T}` - Initial mode probabilities (default: uniform)
"""
function InteractingMultipleModel(
    filters::Vector{F},
    transition_matrix::Matrix{T},
    initial_probs::Vector{T} = fill(T(1.0) / length(filters), length(filters))
) where {T<:Real, F<:AbstractFilter}
    
    M = length(filters)
    @assert M > 0 "Need at least one filter"
    @assert size(transition_matrix) == (M, M) "Transition matrix must be M×M"
    @assert length(initial_probs) == M "Initial probabilities must match number of filters"
    
    # Validate transition matrix (row stochastic)
    for i in 1:M
        @assert abs(sum(transition_matrix[i, :]) - 1.0) < 1e-6 "Each row must sum to 1"
        @assert all(transition_matrix[i, :] .>= 0) "Probabilities must be non-negative"
    end
    
    @assert abs(sum(initial_probs) - 1.0) < 1e-6 "Initial probabilities must sum to 1"
    @assert all(initial_probs .>= 0) "Probabilities must be non-negative"
    
    mixing_probs = zeros(T, M, M)
    likelihoods = ones(T, M)
    predicted_mode_probs = copy(initial_probs)
    
    mixed_states = [state_estimate(f) for f in filters]
    mixed_covs = [covariance(f) for f in filters]
    N = length(mixed_states[1])
    
    # Compute initial overall estimate
    x̂ = compute_overall_estimate(filters, initial_probs)
    P = compute_overall_covariance(filters, initial_probs, x̂)
    
    return InteractingMultipleModel{T, N, F}(
        filters, initial_probs, transition_matrix,
        mixing_probs, likelihoods, predicted_mode_probs,
        mixed_states, mixed_covs, x̂, P
    )
end

"""
    interaction_mixing!(imm::InteractingMultipleModel)

Step 1: Compute mixed initial conditions for each filter.
"""
function interaction_mixing!(imm::InteractingMultipleModel{T, N, F}) where {T, N, F}
    M = length(imm.filters)
    
    # Predicted mode probabilities: c̄ⱼ = ∑ᵢ πᵢⱼ μᵢ(k)
    imm.predicted_mode_probs .= imm.transition_matrix' * imm.mode_probs
    
    # Mixing probabilities: μᵢ|ⱼ(k|k) = πᵢⱼ μᵢ(k) / c̄ⱼ
    for j in 1:M
        for i in 1:M
            if imm.predicted_mode_probs[j] > eps(T)
                imm.mixing_probs[i, j] = imm.transition_matrix[i, j] * 
                                          imm.mode_probs[i] / 
                                          imm.predicted_mode_probs[j]
            else
                # Uniform fallback
                imm.mixing_probs[i, j] = T(1.0) / M
            end
        end
    end
    
    # Compute mixed initial conditions for each filter j
    for j in 1:M
        # Mixed state: x̂⁰ⱼ = ∑ᵢ μᵢ|ⱼ x̂ᵢ
        x̂_mixed = zeros(SVector{N, T})
        for i in 1:M
            x̂ᵢ = state_estimate(imm.filters[i])
            x̂_mixed += imm.mixing_probs[i, j] * x̂ᵢ
        end
        imm.mixed_states[j] = x̂_mixed
        
        # Mixed covariance: P⁰ⱼ = ∑ᵢ μᵢ|ⱼ [Pᵢ + (x̂ᵢ - x̂⁰ⱼ)(x̂ᵢ - x̂⁰ⱼ)']
        P_mixed = zeros(SMatrix{N, N, T})
        for i in 1:M
            x̂ᵢ = state_estimate(imm.filters[i])
            Pᵢ = covariance(imm.filters[i])
            diff = x̂ᵢ - x̂_mixed
            P_mixed += imm.mixing_probs[i, j] * (Pᵢ + diff * diff')
        end
        imm.mixed_covs[j] = make_symmetric(P_mixed)
    end
    
    return imm
end

"""
    model_conditioned_filtering!(imm, models_dyn, models_meas, u, y, dt)

Step 2: Run each model-matched filter with mixed initial conditions.
"""
function model_conditioned_filtering!(
    imm::InteractingMultipleModel{T, N, F},
    models_dyn::Vector,
    models_meas::Vector,
    u::AbstractVector,
    y::AbstractVector,
    dt::T
) where {T, N, F}
    M = length(imm.filters)
    
    for j in 1:M
        # Reinitialize filter with mixed conditions
        set_state!(imm.filters[j], imm.mixed_states[j], imm.mixed_covs[j])
        
        # Run prediction and update
        predict!(imm.filters[j], models_dyn[j], u, dt)
        
        # Compute likelihood before update
        ν_j, S_j = get_predicted_innovation_stats(imm.filters[j], models_meas[j], y)
        
        imm.likelihoods[j] = gaussian_likelihood(ν_j, S_j)
        
        # Update
        update!(imm.filters[j], models_meas[j], y)
    end
    
    return imm
end

"""
    mode_probability_update!(imm)

Step 3: Update mode probabilities using measurement likelihoods.
"""
function mode_probability_update!(imm::InteractingMultipleModel{T, N, F}) where {T, N, F}
    # μⱼ(k+1) ∝ Λⱼ(k+1) * c̄ⱼ
    unnormalized_probs = imm.likelihoods .* imm.predicted_mode_probs
    
    total_prob = sum(unnormalized_probs)
    if total_prob > eps(T)
        imm.mode_probs .= unnormalized_probs ./ total_prob
    else
        # Keep prior if all likelihoods are zero
        imm.mode_probs .= imm.predicted_mode_probs
    end
    
    return imm
end

"""
    estimate_combination!(imm)

Step 4: Combine model-conditioned estimates.
"""
function estimate_combination!(imm::InteractingMultipleModel{T, N, F}) where {T, N, F}
    imm.x̂ = compute_overall_estimate(imm.filters, imm.mode_probs)
    imm.P = compute_overall_covariance(imm.filters, imm.mode_probs, imm.x̂)
    return imm
end

"""
    step!(imm, models_dyn, models_meas, u, y, dt)

Complete IMM cycle: mixing, filtering, update, combination.
"""
function step!(
    imm::InteractingMultipleModel{T, N, F},
    models_dyn::Vector,
    models_meas::Vector,
    u::AbstractVector,
    y::AbstractVector,
    dt::T = T(1.0)
) where {T, N, F}
    
    # Step 1: Interaction/Mixing
    interaction_mixing!(imm)
    
    # Step 2 & 3: Model-conditioned filtering and probability update
    model_conditioned_filtering!(imm, models_dyn, models_meas, u, y, dt)
    mode_probability_update!(imm)
    
    # Step 4: Estimate combination
    estimate_combination!(imm)
    
    return imm
end

"""
    set_state!(filter::AbstractFilter, x, P)

Reinitialize filter state (needs to be defined for each filter type).
"""
function set_state!(filter::KalmanFilter{T, N}, x::SVector{N, T}, P::SMatrix{N, N, T}) where {T, N}
    filter.x = x
    filter.P = P
end

function set_state!(filter::ExtendedKalmanFilter{T, N}, x::SVector{N, T}, P::SMatrix{N, N, T}) where {T, N}
    filter.x = x
    filter.P = P
end

function set_state!(filter::UnscentedKalmanFilter{T, N}, x::SVector{N, T}, P::SMatrix{N, N, T}) where {T, N}
    filter.x = x
    filter.P = P
end

# Add more filter types as needed...

"""
    state_estimate(imm::InteractingMultipleModel)

Return overall MMSE estimate.
"""
function state_estimate(imm::InteractingMultipleModel)
    return imm.x̂
end

"""
    covariance(imm::InteractingMultipleModel)

Return overall covariance.
"""
function covariance(imm::InteractingMultipleModel)
    return imm.P
end

"""
    get_mode_probabilities(imm::InteractingMultipleModel)

Return current mode probabilities.
"""
function get_mode_probabilities(imm::InteractingMultipleModel)
    return imm.mode_probs
end

"""
    most_probable_mode(imm::InteractingMultipleModel)

Return index of mode with highest probability.
"""
function most_probable_mode(imm::InteractingMultipleModel)
    return argmax(imm.mode_probs)
end

"""
    get_model_estimates(imm::InteractingMultipleModel)

Return state estimates from each model-conditioned filter.
"""
function get_model_estimates(imm::InteractingMultipleModel)
    return [state_estimate(f) for f in imm.filters]
end
