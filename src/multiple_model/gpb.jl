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

GPB1 (M filters per step):
1. Start every model-matched filter from the single merged estimate of the
   previous step
2. Predict and update with each model, compute each model's likelihood
3. μⱼ ∝ Λⱼ ∑ᵢ πᵢⱼ μᵢ, then merge into one Gaussian

GPB2 (M² filters per step):
1. Keep one merged estimate per mode from the previous step
2. For every pair (i, j), start from the mode-i estimate and run model j
3. μᵢⱼ ∝ Λᵢⱼ πᵢⱼ μᵢ; merge over i to get one estimate per current mode j

The IMM has the cost of GPB1 and accuracy close to GPB2.

# Reference
- T. Humphreys, Statistical Estimation Theory course notes (UT Austin ASE 381P.6, 2023), §14.2
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

GPB1 filter: every step, all model-matched filters restart from the one
merged estimate of the previous step.
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
    
    likelihoods = zeros(T, M)   # log-likelihoods of the last step
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
    
    logw = zeros(T, M)
    predicted_probs = gpb1.transition_matrix' * gpb1.mode_probs   # ∑ᵢ πᵢⱼ μᵢ
    
    for j in 1:M
        # Every filter starts from the merged estimate of the previous step
        set_state!(gpb1.filters[j], gpb1.x̂, gpb1.P)
        predict!(gpb1.filters[j], models_dyn[j], u, dt)
        
        ν_j, S_j = get_predicted_innovation_stats(gpb1.filters[j], models_meas[j], y)
        gpb1.likelihoods[j] = gaussian_loglikelihood(ν_j, S_j)
        logw[j] = gpb1.likelihoods[j] + log(max(predicted_probs[j], floatmin(T)))
        
        update!(gpb1.filters[j], models_meas[j], y)
    end
    
    gpb1.mode_probs .= normalize_log_weights(logw)
    
    # Merge into a single Gaussian
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

GPB2 filter: keeps one merged estimate per mode and runs a filter for every
(previous mode i, current mode j) pair.
"""
mutable struct GPB2Filter{T<:Real, N, F<:AbstractFilter} <: AbstractFilter
    # Bank of filters for each mode pair (i,j)
    filters::Matrix{F}  # M × M matrix of filters
    
    # Joint probabilities: μᵢⱼ = P(mode_{k-1} = i, mode_k = j | y₁:ₖ)
    joint_probs::Matrix{T}
    
    # Marginal mode probabilities μⱼ(k) and the merged estimate for each mode
    mode_probs::Vector{T}
    mode_states::Vector{SVector{N, T}}
    mode_covs::Vector{SMatrix{N, N, T}}
    
    transition_matrix::Matrix{T}
    likelihoods::Matrix{T}  # log Λᵢⱼ of the last step
    
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
    
    # Before any data, every mode starts from the template's estimate
    mode_probs = copy(initial_probs)
    mode_states = [x0 for _ in 1:M]
    mode_covs = [P0 for _ in 1:M]
    likelihoods = zeros(T, M, M)
    
    x̂ = x0
    P = P0
    
    return GPB2Filter{T, N, F}(filters, joint_probs, mode_probs, mode_states, mode_covs,
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
    
    # 1. For each pair (i, j): start from the mode-i estimate, run model j
    logw = zeros(T, M, M)
    for i in 1:M, j in 1:M
        filt = gpb2.filters[i, j]
        set_state!(filt, gpb2.mode_states[i], gpb2.mode_covs[i])
        predict!(filt, models_dyn[j], u, dt)
        
        ν, S = get_predicted_innovation_stats(filt, models_meas[j], y)
        gpb2.likelihoods[i, j] = gaussian_loglikelihood(ν, S)
        logw[i, j] = gpb2.likelihoods[i, j] +
                     log(max(gpb2.transition_matrix[i, j], floatmin(T))) +
                     log(max(gpb2.mode_probs[i], floatmin(T)))
        
        update!(filt, models_meas[j], y)
    end
    
    # 2. Joint probabilities μᵢⱼ ∝ Λᵢⱼ πᵢⱼ μᵢ and new marginals μⱼ = ∑ᵢ μᵢⱼ
    gpb2.joint_probs = normalize_log_weights(logw)
    gpb2.mode_probs = vec(sum(gpb2.joint_probs, dims=1))
    
    # 3. Merge over the previous mode i: one Gaussian per current mode j
    for j in 1:M
        μj = gpb2.mode_probs[j]
        w = μj > 0 ? gpb2.joint_probs[:, j] ./ μj : fill(T(1) / M, M)
        xj = zeros(SVector{N, T})
        for i in 1:M
            xj += w[i] * state_estimate(gpb2.filters[i, j])
        end
        Pj = zeros(SMatrix{N, N, T})
        for i in 1:M
            d = state_estimate(gpb2.filters[i, j]) - xj
            Pj += w[i] * (covariance(gpb2.filters[i, j]) + d * d')
        end
        gpb2.mode_states[j] = xj
        gpb2.mode_covs[j] = make_symmetric(Pj)
    end
    
    # 4. Overall estimate
    gpb2.x̂ = zeros(SVector{N, T})
    for j in 1:M
        gpb2.x̂ += gpb2.mode_probs[j] * gpb2.mode_states[j]
    end
    P_sum = zeros(SMatrix{N, N, T})
    for j in 1:M
        d = gpb2.mode_states[j] - gpb2.x̂
        P_sum += gpb2.mode_probs[j] * (gpb2.mode_covs[j] + d * d')
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
