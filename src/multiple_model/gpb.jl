"""
    GeneralizedPseudoBayesian{T<:Real, N, F}

Generalized Pseudo-Bayesian (GPB) filters for multiple model estimation.

GPB filters are approximations to the optimal multiple model filter that
reduce computational complexity by pruning the exponentially growing
hypothesis tree.

# GPB Algorithm Family

**GPB0**: No mixing (parallel filters)
**GPB1**: One-step mixing (like IMM but without mode transitions)
**GPB2**: Two-step history

# Mathematical Formulation

The optimal multiple model filter maintains 2^k hypotheses after k steps.
GPB approximations reduce this by:

1. **Merging**: Combine hypotheses with same current mode
2. **Pruning**: Discard low-probability hypotheses

GPB1 Algorithm:
1. Compute model probabilities
2. Run model-matched filters independently
3. Combine estimates (no mixing between filters)

GPB2 Algorithm:
1. Maintain two-step history for each model
2. Merge hypotheses that end in same model pair
3. More accurate than GPB1 but more complex

# Reference
- EstimationNotes.pdf §14.2.1 (GPB Algorithms)
- Ackerson & Fu (1970)
- Blom (1984)
- Bar-Shalom & Li (1993)

# Fields
- `filters::Vector{F}` - Bank of filters
- `mode_probs::Vector{T}` - Mode probabilities
- `transition_matrix::Matrix{T}` - Mode transition probabilities
"""

# ============================================================================
# GPB1 Filter
# ============================================================================

"""
    GPB1Filter{T<:Real, N, F}

GPB1 filter - parallel filters with global mode probability update.

Simpler than IMM: no mixing, each filter runs independently.
"""
mutable struct GPB1Filter{T<:Real, N, F<:AbstractFilter} <: AbstractFilter
    filters::Vector{F}
    mode_probs::Vector{T}
    transition_matrix::Matrix{T}
    likelihoods::Vector{T}
    
    # Overall estimate
    x̂::SVector{N, T}
    P::SMatrix{N, N, T}
end

"""
    GPB1Filter(filters, transition_matrix, initial_probs)

Create GPB1 filter.
"""
function GPB1Filter(
    filters::Vector{F},
    transition_matrix::Matrix{T},
    initial_probs::Vector{T} = fill(T(1.0) / length(filters), length(filters))
) where {T<:Real, F<:AbstractFilter}
    
    M = length(filters)
    @assert size(transition_matrix) == (M, M) "Transition matrix must be M×M"
    @assert length(initial_probs) == M "Initial probabilities must match number of filters"
    
    likelihoods = ones(T, M)
    x̂ = compute_overall_estimate(filters, initial_probs)
    P = compute_overall_covariance(filters, initial_probs, x̂)
    N = length(x̂)
    
    return GPB1Filter{T, N, F}(filters, initial_probs, transition_matrix, likelihoods, x̂, P)
end

"""
    step!(gpb1, models_dyn, models_meas, u, y, dt)

GPB1 filtering step.
"""
function step!(
    gpb1::GPB1Filter{T, N, F},
    models_dyn::Vector,
    models_meas::Vector,
    u::AbstractVector,
    y::AbstractVector,
    dt::T = T(1.0)
) where {T, N, F}
    
    M = length(gpb1.filters)
    
    # 1. Predict each filter independently
    for j in 1:M
        predict!(gpb1.filters[j], models_dyn[j], u, dt)
    end
    
    # 2. Update each filter and compute likelihood
    for j in 1:M
        # Compute innovation before update
        ν_j, S_j = get_predicted_innovation_stats(gpb1.filters[j], models_meas[j], y)
        
        gpb1.likelihoods[j] = gaussian_likelihood(ν_j, S_j)
        
        # Update filter
        update!(gpb1.filters[j], models_meas[j], y)
    end
    
    # 3. Update mode probabilities
    # Predicted mode probs: μ̄_j = ∑ᵢ πᵢⱼ μᵢ
    predicted_probs = gpb1.transition_matrix' * gpb1.mode_probs
    
    # Updated mode probs: μ_j ∝ Λ_j * μ̄_j
    unnorm_probs = gpb1.likelihoods .* predicted_probs
    total = sum(unnorm_probs)
    
    if total > eps(T)
        gpb1.mode_probs .= unnorm_probs / total
    else
        gpb1.mode_probs .= predicted_probs
    end
    
    # 4. Combine estimates
    gpb1.x̂ = compute_overall_estimate(gpb1.filters, gpb1.mode_probs)
    gpb1.P = compute_overall_covariance(gpb1.filters, gpb1.mode_probs, gpb1.x̂)
    
    return gpb1
end

state_estimate(gpb1::GPB1Filter) = gpb1.x̂
covariance(gpb1::GPB1Filter) = gpb1.P
get_mode_probabilities(gpb1::GPB1Filter) = gpb1.mode_probs
most_probable_mode(gpb1::GPB1Filter) = argmax(gpb1.mode_probs)


# ============================================================================
# GPB2 Filter
# ============================================================================

"""
    GPB2Filter{T<:Real, N, F}

GPB2 filter - maintains two-step mode history.

More accurate than GPB1 by tracking mode sequences (i,j) where i is previous mode.
"""
mutable struct GPB2Filter{T<:Real, N, F<:AbstractFilter} <: AbstractFilter
    # Bank of filters for each mode pair (i,j)
    filters::Matrix{F}  # M × M matrix of filters
    
    # Joint probabilities: μᵢⱼ = P(mode_k = j, mode_{k-1} = i | y₁:ₖ)
    joint_probs::Matrix{T}
    
    # Marginal mode probabilities
    mode_probs::Vector{T}
    
    transition_matrix::Matrix{T}
    likelihoods::Vector{T}
    
    # Overall estimate
    x̂::SVector{N, T}
    P::SMatrix{N, N, T}
    
    M::Int  # Number of modes
end

"""
    GPB2Filter(filter_template, transition_matrix, initial_probs)

Create GPB2 filter by cloning template filter.
"""
function GPB2Filter(
    filter_template::F,
    M::Int,
    transition_matrix::Matrix{T},
    initial_probs::Vector{T} = fill(T(1.0) / M, M)
) where {T<:Real, F<:AbstractFilter}
    
    @assert size(transition_matrix) == (M, M) "Transition matrix must be M×M"
    @assert length(initial_probs) == M "Initial probabilities must match M"
    
    # Create M×M filter bank
    filters = Matrix{F}(undef, M, M)
    x0 = state_estimate(filter_template)
    P0 = covariance(filter_template)
    N = length(x0)
    
    for i in 1:M
        for j in 1:M
            # Clone filter for mode pair (i,j)
            filters[i,j] = Base.typename(F).wrapper(x0, P0)
        end
    end
    
    # Initialize joint probabilities: μᵢⱼ = πᵢⱼ * μᵢ
    joint_probs = zeros(T, M, M)
    for i in 1:M
        for j in 1:M
            joint_probs[i,j] = transition_matrix[i,j] * initial_probs[i]
        end
    end
    joint_probs ./= sum(joint_probs)
    
    mode_probs = vec(sum(joint_probs, dims=1))
    likelihoods = ones(T, M)
    
    x̂ = x0
    P = P0
    
    return GPB2Filter{T, N, F}(filters, joint_probs, mode_probs, 
                                transition_matrix, likelihoods, x̂, P, M)
end

"""
    step!(gpb2, models_dyn, models_meas, u, y, dt)

GPB2 filtering step with two-step history.
"""
function step!(
    gpb2::GPB2Filter{T, N, F},
    models_dyn::Vector,
    models_meas::Vector,
    u::AbstractVector,
    y::AbstractVector,
    dt::T = T(1.0)
) where {T, N, F}
    
    M = gpb2.M
    
    # 1. Predict each filter in the bank
    for i in 1:M
        for j in 1:M
            predict!(gpb2.filters[i,j], models_dyn[j], u, dt)
        end
    end
    
    # 2. Update filters and compute likelihoods for each current mode j
    for j in 1:M
        # For mode j, we need to merge estimates from all previous modes i
        # Compute likelihood for mode j (using merged estimate)
        
        # Get marginal probability for mode j at k-1
        μj_prev = sum(gpb2.joint_probs[:, j])
        
        if μj_prev > eps(T)
            # Weighted merge of filters ending in mode j
            x̂_merged = zeros(SVector{N, T})
            for i in 1:M
                weight = gpb2.joint_probs[i,j] / μj_prev
                x̂_merged += weight * state_estimate(gpb2.filters[i,j])
            end
            
            ŷ = measure(models_meas[j], x̂_merged)
            
            # Use first filter's innovation covariance (approximation)
            _, S = get_predicted_innovation_stats(gpb2.filters[1,j], models_meas[j], y)
            ν = y - ŷ
            
            gpb2.likelihoods[j] = gaussian_likelihood(ν, S)
        else
            gpb2.likelihoods[j] = T(1.0)
        end
        
        # Update all filters ending in mode j
        for i in 1:M
            update!(gpb2.filters[i,j], models_meas[j], y)
        end
    end
    
    # 3. Update joint probabilities
    # μᵢⱼ(k) ∝ Λⱼ(k) * πᵢⱼ * μᵢ(k-1)
    new_joint_probs = zeros(T, M, M)
    μ_prev = vec(sum(gpb2.joint_probs, dims=2))  # Previous marginals
    
    for i in 1:M
        for j in 1:M
            new_joint_probs[i,j] = gpb2.likelihoods[j] * 
                                   gpb2.transition_matrix[i,j] * 
                                   μ_prev[i]
        end
    end
    
    total = sum(new_joint_probs)
    if total > eps(T)
        new_joint_probs ./= total
    end
    gpb2.joint_probs = new_joint_probs
    
    # Update marginal mode probabilities
    gpb2.mode_probs = vec(sum(gpb2.joint_probs, dims=1))
    
    # 4. Combine estimates across all mode pairs
    gpb2.x̂ = zeros(SVector{N, T})
    for i in 1:M
        for j in 1:M
            gpb2.x̂ += gpb2.joint_probs[i,j] * state_estimate(gpb2.filters[i,j])
        end
    end
    
    # Covariance
    P_sum = zeros(SMatrix{N, N, T})
    for i in 1:M
        for j in 1:M
            x̂ᵢⱼ = state_estimate(gpb2.filters[i,j])
            Pᵢⱼ = covariance(gpb2.filters[i,j])
            diff = x̂ᵢⱼ - gpb2.x̂
            P_sum += gpb2.joint_probs[i,j] * (Pᵢⱼ + diff * diff')
        end
    end
    gpb2.P = make_symmetric(P_sum)
    
    return gpb2
end

state_estimate(gpb2::GPB2Filter) = gpb2.x̂
covariance(gpb2::GPB2Filter) = gpb2.P
get_mode_probabilities(gpb2::GPB2Filter) = gpb2.mode_probs
most_probable_mode(gpb2::GPB2Filter) = argmax(gpb2.mode_probs)

"""
    get_joint_probabilities(gpb2::GPB2Filter)

Return joint mode probabilities μᵢⱼ.
"""
function get_joint_probabilities(gpb2::GPB2Filter)
    return gpb2.joint_probs
end
