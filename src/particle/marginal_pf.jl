"""
    MarginalParticleFilter{T<:Real, N}

Marginal Particle Filter (MPF) - direct marginal estimation.

The MPF directly estimates marginal distributions of state components without
computing the full joint posterior. This is useful when only certain state
components are of interest, or when the full state is high-dimensional.

# Mathematical Formulation

Instead of approximating joint p(x₁, x₂, ..., xₙ | y₁:ₖ), directly estimate
marginals:
```math
p(xᵢ | y₁:ₖ) for i = 1, 2, ..., n
```

This can be more efficient than computing the full joint and then marginalizing.

# Algorithm

1. Maintain particles for each marginal distribution independently
2. Use auxiliary variables or conditional sampling to maintain dependencies
3. Weight particles based on marginal likelihoods

# Advantages
- Lower-dimensional particle representation
- Can focus on specific state components
- More efficient for certain problems

# Limitations
- Loses cross-correlations between components
- Requires careful handling of dependencies
- Not suitable when full state estimate is needed

# Reference
- R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 3: Nonlinear Estimation (UT Austin ASE 381P.7, 2023), §5.9
- Klaas et al. (2005)

# Fields
- `marginal_particles::Vector{Vector{T}}` - Particles for each marginal (one vector per dimension)
- `marginal_weights::Vector{Vector{T}}` - Weights for each marginal
- `N_particles::Int` - Number of particles per marginal
"""
mutable struct MarginalParticleFilter{T<:Real, N} <: AbstractFilter
    # Particles for each state dimension independently
    marginal_particles::Vector{Vector{T}}  # N vectors, each with N_particles elements
    marginal_weights::Vector{Vector{T}}
    N_particles::Int
    
    # Resampling
    resample_threshold::T
    resample_method::Symbol
    rng::AbstractRNG
    
    # ESS for each marginal
    ess::Vector{T}
end

"""
    MarginalParticleFilter(μ0, P0, N_particles; kwargs...)

Create Marginal Particle Filter.

# Arguments
- `μ0::SVector{N,T}` - Initial mean
- `P0::SMatrix{N,N,T}` - Initial covariance
- `N_particles::Int` - Number of particles per marginal

# Keyword Arguments
- `resample_threshold::T = 0.5` - ESS threshold
- `resample_method::Symbol = :systematic`
- `rng::AbstractRNG = Random.GLOBAL_RNG`
"""
function MarginalParticleFilter(
    μ0::SVector{N, T},
    P0::SMatrix{N, N, T},
    N_particles::Int;
    resample_threshold::T = T(0.5),
    resample_method::Symbol = :systematic,
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {T<:Real, N}
    
    @assert N_particles > 0 "Need at least one particle"
    
    # Initialize particles for each marginal
    marginal_particles = Vector{Vector{T}}(undef, N)
    marginal_weights = Vector{Vector{T}}(undef, N)
    
    for i in 1:N
        # Sample from marginal: xᵢ ~ N(μ0[i], P0[i,i])
        σᵢ = sqrt(P0[i,i])
        marginal_particles[i] = μ0[i] .+ σᵢ * randn(rng, T, N_particles)
        marginal_weights[i] = fill(T(1.0) / N_particles, N_particles)
    end
    
    ess = fill(T(N_particles), N)
    
    return MarginalParticleFilter{T, N}(
        marginal_particles, marginal_weights, N_particles,
        resample_threshold, resample_method, rng, ess
    )
end

"""
    predict!(mpf::MarginalParticleFilter, model_dyn, u, dt)

Prediction step for marginal particles.

Note: This simplified version assumes weak coupling between dimensions.
For strongly coupled dynamics, more sophisticated approaches are needed.
"""
function predict!(
    mpf::MarginalParticleFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    u::AbstractVector = zeros(T, 0),
    dt::T = T(1.0)
) where {T, N}
    
    Q = process_noise_cov(model_dyn, dt)
    
    # For each dimension
    for dim in 1:N
        for p in 1:mpf.N_particles
            # Reconstruct full state from marginal samples
            # (Simplified: use means for other dimensions)
            x_full = SVector{N, T}([dim == d ? mpf.marginal_particles[d][p] : 
                                     Statistics.mean(mpf.marginal_particles[d]) 
                                     for d in 1:N])
            
            # Propagate
            x_next = propagate(model_dyn, x_full, u, dt)
            
            # Add process noise to this dimension
            mpf.marginal_particles[dim][p] = x_next[dim] + sqrt(Q[dim,dim]) * randn(mpf.rng, T)
        end
    end
    
    return mpf
end

"""
    update!(mpf::MarginalParticleFilter, model_meas, y)

Measurement update for marginal particles.
"""
function update!(
    mpf::MarginalParticleFilter{T, N},
    model_meas::AbstractMeasurementModel,
    y::AbstractVector
) where {T, N}
    
    R = measurement_noise_cov(model_meas)
    
    # Update each marginal
    for dim in 1:N
        for p in 1:mpf.N_particles
            # Reconstruct full state
            x_full = SVector{N, T}([dim == d ? mpf.marginal_particles[d][p] : 
                                     Statistics.mean(mpf.marginal_particles[d]) 
                                     for d in 1:N])
            
            # Predicted measurement
            ŷ = measure(model_meas, x_full)
            innov = y - ŷ
            
            # Update weight (likelihood)
            log_like = -0.5 * dot(innov, R \ innov)
            mpf.marginal_weights[dim][p] *= exp(log_like)
        end
        
        # Normalize weights for this dimension
        total = sum(mpf.marginal_weights[dim])
        if total > eps(T)
            mpf.marginal_weights[dim] ./= total
        else
            mpf.marginal_weights[dim] .= T(1.0) / mpf.N_particles
        end
        
        # Compute ESS
        mpf.ess[dim] = effective_sample_size(mpf.marginal_weights[dim])
        
        # Resample if needed
        if mpf.ess[dim] < mpf.resample_threshold * mpf.N_particles
            if mpf.resample_method == :systematic
                indices = systematic_resample(mpf.marginal_weights[dim]; rng=mpf.rng)
            elseif mpf.resample_method == :stratified
                indices = stratified_resample(mpf.marginal_weights[dim]; rng=mpf.rng)
            else
                indices = multinomial_resample(mpf.marginal_weights[dim]; rng=mpf.rng)
            end
            
            mpf.marginal_particles[dim] = mpf.marginal_particles[dim][indices]
            mpf.marginal_weights[dim] .= T(1.0) / mpf.N_particles
            mpf.ess[dim] = T(mpf.N_particles)
        end
    end
    
    return mpf
end

"""
    step!(mpf, model_dyn, model_meas, u, y, dt)

Combined predict and update.
"""
function step!(
    mpf::MarginalParticleFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    model_meas::AbstractMeasurementModel,
    u::AbstractVector,
    y::AbstractVector,
    dt::T = T(1.0)
) where {T, N}
    
    predict!(mpf, model_dyn, u, dt)
    update!(mpf, model_meas, y)
    return mpf
end

"""
    state_estimate(mpf::MarginalParticleFilter)

Compute MMSE estimate from marginals.
"""
function state_estimate(mpf::MarginalParticleFilter{T, N}) where {T, N}
    x̂ = zeros(SVector{N, T})
    
    for dim in 1:N
        # Weighted mean of marginal
        x̂_dim = sum(mpf.marginal_weights[dim] .* mpf.marginal_particles[dim])
        x̂ = setindex(x̂, x̂_dim, dim)
    end
    
    return x̂
end

"""
    covariance(mpf::MarginalParticleFilter)

Compute marginal covariance (diagonal, no cross-terms).
"""
function covariance(mpf::MarginalParticleFilter{T, N}) where {T, N}
    x̂ = state_estimate(mpf)
    P = zeros(SMatrix{N, N, T})
    
    for dim in 1:N
        # Marginal variance
        var_dim = sum(mpf.marginal_weights[dim] .* 
                     (mpf.marginal_particles[dim] .- x̂[dim]).^2)
        P = setindex(P, var_dim, dim, dim)
    end
    
    return P
end

"""
    get_marginal_estimate(mpf::MarginalParticleFilter, dim::Int)

Get estimate for specific dimension.
"""
function get_marginal_estimate(mpf::MarginalParticleFilter{T}, dim::Int) where {T}
    return sum(mpf.marginal_weights[dim] .* mpf.marginal_particles[dim])
end

"""
    get_marginal_variance(mpf::MarginalParticleFilter, dim::Int)

Get variance for specific dimension.
"""
function get_marginal_variance(mpf::MarginalParticleFilter{T}, dim::Int) where {T}
    μ = get_marginal_estimate(mpf, dim)
    return sum(mpf.marginal_weights[dim] .* (mpf.marginal_particles[dim] .- μ).^2)
end

"""
    get_marginal_particles(mpf::MarginalParticleFilter, dim::Int)

Return particles and weights for specific dimension.
"""
function get_marginal_particles(mpf::MarginalParticleFilter, dim::Int)
    return mpf.marginal_particles[dim], mpf.marginal_weights[dim]
end

"""
    num_particles(mpf::MarginalParticleFilter)

Return number of particles per marginal.
"""
function num_particles(mpf::MarginalParticleFilter)
    return mpf.N_particles
end

"""
    effective_particles(mpf::MarginalParticleFilter, dim::Int)

Return ESS for specific dimension.
"""
function effective_particles(mpf::MarginalParticleFilter, dim::Int)
    return mpf.ess[dim]
end
