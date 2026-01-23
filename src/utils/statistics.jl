# Statistical utility functions

"""
    sample_mean(particles, weights)

Compute weighted sample mean.

# Arguments
- `particles`: Vector of particles (each is a state vector)
- `weights`: Vector of weights (should sum to 1)

Returns the weighted mean state.
"""
function sample_mean(particles::Vector{<:AbstractVector{T}}, weights::Vector{T}) where {T}
    @assert length(particles) == length(weights)
    @assert isapprox(sum(weights), 1.0, atol=1e-8) "Weights should sum to 1"
    
    # Weighted sum
    μ = sum(w * x for (w, x) in zip(weights, particles))
    return μ
end

"""
    sample_covariance(particles, weights, mean=nothing)

Compute weighted sample covariance.

# Arguments
- `particles`: Vector of particles
- `weights`: Vector of weights (should sum to 1)
- `mean`: Pre-computed mean (optional, computed if not provided)

Returns the weighted covariance matrix.
"""
function sample_covariance(particles::Vector{<:AbstractVector{T}}, weights::Vector{T}, 
                          mean=nothing) where {T}
    @assert length(particles) == length(weights)
    
    if isnothing(mean)
        mean = sample_mean(particles, weights)
    end
    
    n = length(particles[1])
    Σ = zeros(T, n, n)
    
    for (w, x) in zip(weights, particles)
        diff = x - mean
        Σ += w * (diff * diff')
    end
    
    return Σ
end

"""
    effective_sample_size(weights)

Compute the effective sample size (ESS) of a particle set.

ESS = 1 / Σ(w_i²)

ESS ranges from 1 (all weight on one particle) to N (uniform weights).
Low ESS indicates weight degeneracy.
"""
function effective_sample_size(weights::AbstractVector)
    return 1.0 / sum(w^2 for w in weights)
end

"""
    weighted_percentile(values, weights, p)

Compute the p-th percentile of weighted samples.

# Arguments
- `values`: Sample values
- `weights`: Sample weights (should sum to 1)
- `p`: Percentile (0 to 1)
"""
function weighted_percentile(values::AbstractVector, weights::AbstractVector, p::Real)
    @assert 0 ≤ p ≤ 1 "Percentile must be between 0 and 1"
    @assert length(values) == length(weights)
    
    # Sort by value
    idx = sortperm(values)
    sorted_vals = values[idx]
    sorted_weights = weights[idx]
    
    # Cumulative sum of weights
    cumsum_w = cumsum(sorted_weights)
    
    # Find the index where cumsum exceeds p
    i = findfirst(cw -> cw ≥ p, cumsum_w)
    
    if isnothing(i)
        return sorted_vals[end]
    else
        return sorted_vals[i]
    end
end

"""
    normalize_log_weights(log_weights)

Numerically stable normalization of log weights.

Prevents underflow/overflow by using the log-sum-exp trick.

Returns normalized weights (not in log space).
"""
function normalize_log_weights(log_weights::AbstractVector{T}) where {T}
    # Log-sum-exp trick for numerical stability
    max_log_w = maximum(log_weights)
    weights = exp.(log_weights .- max_log_w)
    return weights ./ sum(weights)
end
