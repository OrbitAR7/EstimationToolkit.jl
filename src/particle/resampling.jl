# Resampling algorithms for particle filters

"""
    systematic_resample(weights, N=length(weights); rng=Random.GLOBAL_RNG)

Systematic resampling of particles.

Systematic resampling is deterministic given a random starting point, which
reduces sampling variance compared to multinomial resampling.

# Arguments
- `weights`: Normalized particle weights
- `N`: Number of particles to resample (default: same as input)
- `rng`: Random number generator

# Returns
Vector of indices into the original particle array.

# References
- 03-Nonlinear_filters.pdf, Section 5
- Arulampalam et al. (2002)
"""
function systematic_resample(weights::AbstractVector{T}, N=length(weights);
                            rng=Random.GLOBAL_RNG) where {T}
    @assert isapprox(sum(weights), 1.0, atol=1e-8) "Weights must sum to 1"
    
    indices = Vector{Int}(undef, N)
    
    # Generate systematic samples
    u0 = rand(rng, T) / N
    cumsum_w = cumsum(weights)
    
    j = 1
    for i in 1:N
        u = u0 + (i - 1) / N
        
        # Find index where cumsum exceeds u
        while j <= length(cumsum_w) && cumsum_w[j] < u
            j += 1
        end
        
        indices[i] = min(j, length(weights))
    end
    
    return indices
end

"""
    stratified_resample(weights, N=length(weights); rng=Random.GLOBAL_RNG)

Stratified resampling of particles.

Similar to systematic resampling but samples independently within each stratum,
providing better coverage of the weight distribution.

# Arguments
- `weights`: Normalized particle weights
- `N`: Number of particles to resample
- `rng`: Random number generator

# Returns
Vector of indices into the original particle array.
"""
function stratified_resample(weights::AbstractVector{T}, N=length(weights);
                            rng=Random.GLOBAL_RNG) where {T}
    @assert isapprox(sum(weights), 1.0, atol=1e-8) "Weights must sum to 1"
    
    indices = Vector{Int}(undef, N)
    cumsum_w = cumsum(weights)
    
    j = 1
    for i in 1:N
        # Random sample in stratum i
        u = (i - 1 + rand(rng, T)) / N
        
        # Find index
        while j <= length(cumsum_w) && cumsum_w[j] < u
            j += 1
        end
        
        indices[i] = min(j, length(weights))
        j = 1  # Reset for next stratum
    end
    
    return indices
end

"""
    multinomial_resample(weights, N=length(weights); rng=Random.GLOBAL_RNG)

Multinomial (random) resampling of particles.

Simple but has highest variance among common resampling schemes.
Included mainly for comparison purposes.

# Arguments
- `weights`: Normalized particle weights
- `N`: Number of particles to resample
- `rng`: Random number generator

# Returns
Vector of indices into the original particle array.
"""
function multinomial_resample(weights::AbstractVector{T}, N=length(weights);
                             rng=Random.GLOBAL_RNG) where {T}
    @assert isapprox(sum(weights), 1.0, atol=1e-8) "Weights must sum to 1"
    
    # Sample from categorical distribution N times
    cumsum_w = cumsum(weights)
    indices = Vector{Int}(undef, N)
    
    for i in 1:N
        u = rand(rng, T)
        # Find first index where cumsum exceeds u
        idx = findfirst(c -> c >= u, cumsum_w)
        indices[i] = isnothing(idx) ? length(weights) : idx
    end
    
    return indices
end

"""
    resample!(particles, weights, method=:systematic; 
              threshold=0.5, rng=Random.GLOBAL_RNG)

In-place resampling of particles when effective sample size is low.

# Arguments
- `particles`: Vector of particles (modified in-place)
- `weights`: Vector of weights (reset to uniform after resampling)
- `method`: Resampling method (:systematic, :stratified, or :multinomial)
- `threshold`: Resample when ESS/N < threshold
- `rng`: Random number generator

# Returns
Boolean indicating whether resampling occurred.
"""
function resample!(particles::Vector, weights::Vector{T}, method=:systematic;
                   threshold=0.5, rng=Random.GLOBAL_RNG) where {T}
    N = length(particles)
    ess = effective_sample_size(weights)
    
    # Only resample if ESS is too low
    if ess / N >= threshold
        return false
    end
    
    # Choose resampling method
    if method == :systematic
        indices = systematic_resample(weights, N; rng=rng)
    elseif method == :stratified
        indices = stratified_resample(weights, N; rng=rng)
    elseif method == :multinomial
        indices = multinomial_resample(weights, N; rng=rng)
    else
        error("Unknown resampling method: $method")
    end
    
    # Resample particles
    particles[:] = particles[indices]
    
    # Reset weights to uniform
    weights[:] .= 1.0 / N
    
    return true
end
