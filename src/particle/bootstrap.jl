"""
    BootstrapParticleFilter{T<:Real, N}

Bootstrap particle filter (also known as SIR - Sequential Importance Resampling).

The bootstrap PF uses the transition density as the proposal distribution,
making it simple to implement but sometimes inefficient for sharp measurement
likelihoods.

System model:
- Dynamics: x_{k+1} = f(x_k, u_k) + v_k
- Measurement: y_k ~ p(y|x_k)
- Proposal: x_k^(i) ~ p(x_k | x_{k-1}^(i))

# Fields
- `particles::Vector{SVector{N, T}}`: Particle states
- `weights::Vector{T}`: Normalized particle weights
- `resample_method::Symbol`: Resampling algorithm (:systematic, :stratified, :multinomial)
- `resample_threshold::T`: Resample when ESS/N < threshold
- `rng::AbstractRNG`: Random number generator

# References
- R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 3: Nonlinear Estimation (UT Austin ASE 381P.7, 2023), §5.1
- R. Zanetti, "Sequential Importance Sampling," ASE 381P.7 lecture slides (UT Austin, 2023)
- Gordon, Salmond & Smith (1993)
"""
mutable struct BootstrapParticleFilter{T<:Real, N} <: AbstractFilter
    particles::Vector{SVector{N, T}}
    weights::Vector{T}
    
    # Resampling configuration
    resample_method::Symbol
    resample_threshold::T
    
    # RNG for reproducibility
    rng::AbstractRNG
    
    # Store last measurement for diagnostics
    last_measurement::Union{Nothing, SVector}
    
    function BootstrapParticleFilter(particles::Vector{SVector{N, T}},
                                     weights::Vector{T}=fill(1.0/length(particles), length(particles));
                                     resample_method=:systematic,
                                     resample_threshold=0.5,
                                     rng=Random.GLOBAL_RNG) where {N, T<:Real}
        @assert length(particles) == length(weights)
        @assert resample_method in [:systematic, :stratified, :multinomial]
        @assert 0 < resample_threshold ≤ 1
        
        # Normalize weights
        w_sum = sum(weights)
        if !isapprox(w_sum, 1.0, atol=1e-8)
            weights = weights ./ w_sum
        end
        
        new{T, N}(particles, weights, resample_method, resample_threshold, rng, nothing)
    end
end

# Convenience constructor with initial distribution
function BootstrapParticleFilter(μ0::AbstractVector{T}, P0::AbstractMatrix{T}, N_particles::Int;
                                 kwargs...) where {T<:Real}
    n = length(μ0)
    
    # Sample initial particles from Gaussian
    L = cholesky(Symmetric(P0)).L
    particles = [SVector{n, T}(μ0 + L * randn(n)) for _ in 1:N_particles]
    weights = fill(1.0 / N_particles, N_particles)
    
    return BootstrapParticleFilter(particles, weights; kwargs...)
end

"""
    predict!(pf::BootstrapParticleFilter, model::AbstractDynamicsModel, u, dt)

Particle filter prediction step.

Propagates each particle through the dynamics model with sampled process noise:
x_k^(i) ~ p(x_k | x_{k-1}^(i)) = f(x_{k-1}^(i), u, dt) + v_k

Weights remain unchanged during prediction (in bootstrap PF).
"""
function predict!(pf::BootstrapParticleFilter{T, N}, model::AbstractDynamicsModel,
                  u, dt) where {T, N}
    # Propagate each particle
    for i in eachindex(pf.particles)
        # Propagate through dynamics
        x_pred = propagate(model, pf.particles[i], u, dt)
        
        # Add process noise
        v = sample_process_noise(model, dt, pf.rng)
        pf.particles[i] = x_pred + SVector{N, T}(v)
    end
    
    return pf
end

"""
    update!(pf::BootstrapParticleFilter, model::AbstractMeasurementModel, y)

Particle filter measurement update.

Updates particle weights based on measurement likelihood:
w_k^(i) ∝ w_{k-1}^(i) * p(y_k | x_k^(i))

Resamples if effective sample size drops below threshold.
"""
function update!(pf::BootstrapParticleFilter{T, N}, model::AbstractMeasurementModel,
                 y) where {T, N}
    # Update weights using measurement likelihood
    for i in eachindex(pf.particles)
        likelihood_val = likelihood(model, y, pf.particles[i])
        pf.weights[i] *= likelihood_val
    end
    
    # Normalize weights
    w_sum = sum(pf.weights)
    if w_sum > 0
        pf.weights ./= w_sum
    else
        # All weights are zero - reset to uniform (degeneracy)
        @warn "Particle filter degeneracy: all weights near zero"
        pf.weights .= 1.0 / length(pf.weights)
    end
    
    # Resample if ESS is low
    ess = effective_sample_size(pf.weights)
    if ess / length(pf.particles) < pf.resample_threshold
        resample!(pf.particles, pf.weights, pf.resample_method; rng=pf.rng)
    end
    
    pf.last_measurement = y
    
    return pf
end

"""
    state_estimate(pf::BootstrapParticleFilter)

Compute the weighted mean of particles as the state estimate.
"""
function state_estimate(pf::BootstrapParticleFilter{T, N}) where {T, N}
    μ = sum(w * x for (w, x) in zip(pf.weights, pf.particles))
    return SVector{N, T}(μ)
end

"""
    covariance(pf::BootstrapParticleFilter)

Compute the weighted sample covariance of particles.
"""
function covariance(pf::BootstrapParticleFilter{T, N}) where {T, N}
    μ = state_estimate(pf)
    
    Σ = zeros(T, N, N)
    for (w, x) in zip(pf.weights, pf.particles)
        dx = x - μ
        Σ += w * (dx * dx')
    end
    
    return SMatrix{N, N, T}(Σ)
end

"""
    num_particles(pf::BootstrapParticleFilter)

Return the number of particles.
"""
num_particles(pf::BootstrapParticleFilter) = length(pf.particles)

"""
    effective_particles(pf::BootstrapParticleFilter)

Return the effective sample size.
"""
effective_particles(pf::BootstrapParticleFilter) = effective_sample_size(pf.weights)

# Convert to/from ParticleBelief
BootstrapParticleFilter(belief::ParticleBelief{T, N}; kwargs...) where {T, N} =
    BootstrapParticleFilter(belief.particles, belief.weights; kwargs...)

ParticleBelief(pf::BootstrapParticleFilter) = 
    ParticleBelief(pf.particles, pf.weights)
