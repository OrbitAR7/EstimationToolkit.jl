"""
    GaussianParticleFilter{T<:Real, N}

Gaussian Particle Filter (GPF) - Hybrid particle/Gaussian filter.

The GPF approximates the posterior distribution as a Gaussian, using particles
to compute the mean and covariance at each step. This reduces the particle
degeneracy problem compared to pure particle filters while retaining the ability
to handle nonlinear systems.

# Mathematical Formulation

Instead of representing posterior as weighted particles, approximate as:
```math
p(xₖ | y₁:ₖ) ≈ N(x̂ₖ, Pₖ)
```

where x̂ₖ and Pₖ are computed from weighted particles:
```math
x̂ₖ = ∑ᵢ wᵢ⁽ᵏ⁾ xᵢ⁽ᵏ⁾
Pₖ = ∑ᵢ wᵢ⁽ᵏ⁾ (xᵢ⁽ᵏ⁾ - x̂ₖ)(xᵢ⁽ᵏ⁾ - x̂ₖ)ᵀ
```

At each step:
1. Sample particles from current Gaussian: xᵢ ~ N(x̂ₖ₋₁, Pₖ₋₁)
2. Propagate through dynamics
3. Update weights using measurement likelihood
4. Fit Gaussian to weighted particles
5. Resample if needed

# Advantages
- Smooth Gaussian approximation
- Reduced particle degeneracy
- Easier to work with downstream (Kalman-like output)

# Limitations
- Inappropriate for truly multi-modal posteriors
- Gaussian assumption may be restrictive

# Reference
- R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 3: Nonlinear Estimation (UT Austin ASE 381P.7, 2023), §5.3
- Kotecha & Djuric (2003)

# Fields
- `x̂::SVector{N,T}` - Current Gaussian mean
- `P::SMatrix{N,N,T}` - Current Gaussian covariance
- `particles::Vector{SVector{N,T}}` - Particle locations
- `weights::Vector{T}` - Particle weights
- `N_particles::Int` - Number of particles
"""
mutable struct GaussianParticleFilter{T<:Real, N} <: AbstractFilter
    x̂::SVector{N, T}
    P::SMatrix{N, N, T}
    
    particles::Vector{SVector{N, T}}
    weights::Vector{T}
    N_particles::Int
    
    # Resampling
    resample_threshold::T
    resample_method::Symbol
    rng::AbstractRNG
    
    # Storage
    ess::T
end

"""
    GaussianParticleFilter(x0, P0, N_particles; kwargs...)

Create a Gaussian Particle Filter.

# Arguments
- `x0::SVector{N,T}` - Initial mean
- `P0::SMatrix{N,N,T}` - Initial covariance
- `N_particles::Int` - Number of particles

# Keyword Arguments
- `resample_threshold::T = 0.5` - ESS threshold for resampling (fraction of N)
- `resample_method::Symbol = :systematic` - Resampling method
- `rng::AbstractRNG = Random.GLOBAL_RNG` - Random number generator
"""
function GaussianParticleFilter(
    x0::SVector{N, T},
    P0::SMatrix{N, N, T},
    N_particles::Int;
    resample_threshold::T = T(0.5),
    resample_method::Symbol = :systematic,
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {T<:Real, N}
    
    @assert N_particles > 0 "Need at least one particle"
    @assert is_positive_definite(P0) "Initial covariance must be SPD"
    
    # Initialize particles by sampling from initial Gaussian
    particles = sample_gaussian(x0, P0, N_particles, rng)
    weights = fill(T(1.0) / N_particles, N_particles)
    
    ess = T(N_particles)
    
    return GaussianParticleFilter{T, N}(
        x0, P0, particles, weights, N_particles,
        resample_threshold, resample_method, rng, ess
    )
end

"""
    sample_gaussian(μ, Σ, N, rng)

Sample N particles from Gaussian N(μ, Σ).
"""
function sample_gaussian(
    μ::SVector{N, T},
    Σ::SMatrix{N, N, T},
    N_samples::Int,
    rng::AbstractRNG
) where {N, T}
    
    # Cholesky decomposition: Σ = L * L'
    L = cholesky(Symmetric(Σ)).L
    
    particles = Vector{SVector{N, T}}(undef, N_samples)
    
    for i in 1:N_samples
        # Sample standard normal
        z = SVector{N, T}(randn(rng, T, N))
        # Transform: x = μ + L * z
        particles[i] = μ + L * z
    end
    
    return particles
end

"""
    predict!(gpf::GaussianParticleFilter, model_dyn, u, dt)

Prediction: sample particles from current Gaussian and propagate.
"""
function predict!(
    gpf::GaussianParticleFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    u::AbstractVector = zeros(T, 0),
    dt::T = T(1.0)
) where {T, N}
    
    # Sample particles from current Gaussian approximation
    gpf.particles = sample_gaussian(gpf.x̂, gpf.P, gpf.N_particles, gpf.rng)
    
    # Propagate each particle through dynamics
    for i in eachindex(gpf.particles)
        # Add process noise
        v = sample_process_noise(model_dyn, dt, gpf.rng)
        x_next = propagate(model_dyn, gpf.particles[i], u, dt)
        gpf.particles[i] = x_next + v
    end
    
    # Compute predicted Gaussian from propagated particles
    gpf.x̂ = sample_mean(gpf.particles, gpf.weights)
    gpf.P = sample_covariance(gpf.particles, gpf.weights, gpf.x̂)
    gpf.P = ensure_spd(gpf.P)
    
    return gpf
end

"""
    update!(gpf::GaussianParticleFilter, model_meas, y)

Measurement update: reweight particles and fit Gaussian.
"""
function update!(
    gpf::GaussianParticleFilter{T, N},
    model_meas::AbstractMeasurementModel,
    y::AbstractVector
) where {T, N}
    
    R = measurement_noise_cov(model_meas)
    
    # Update weights using measurement likelihood
    for i in eachindex(gpf.particles)
        ŷ = measure(model_meas, gpf.particles[i])
        innov = y - ŷ
        
        # Gaussian likelihood: exp(-0.5 * innov' * R^-1 * innov)
        log_like = -0.5 * dot(innov, R \ innov)
        gpf.weights[i] *= exp(log_like)
    end
    
    # Normalize weights
    total_weight = sum(gpf.weights)
    if total_weight > eps(T)
        gpf.weights ./= total_weight
    else
        # Uniform fallback if all weights collapsed
        gpf.weights .= T(1.0) / gpf.N_particles
    end
    
    # Compute ESS
    gpf.ess = effective_sample_size(gpf.weights)
    
    # Fit Gaussian to weighted particles
    gpf.x̂ = sample_mean(gpf.particles, gpf.weights)
    gpf.P = sample_covariance(gpf.particles, gpf.weights, gpf.x̂)
    gpf.P = ensure_spd(gpf.P)
    
    # Resample if needed
    if gpf.ess < gpf.resample_threshold * gpf.N_particles
        resample!(gpf)
    end
    
    return gpf
end

"""
    resample!(gpf::GaussianParticleFilter)

Resample particles from current Gaussian approximation.
"""
function resample!(gpf::GaussianParticleFilter{T, N}) where {T, N}
    # Simply resample from current Gaussian (regularization)
    gpf.particles = sample_gaussian(gpf.x̂, gpf.P, gpf.N_particles, gpf.rng)
    gpf.weights .= T(1.0) / gpf.N_particles
    gpf.ess = T(gpf.N_particles)
    return gpf
end

"""
    step!(gpf, model_dyn, model_meas, u, y, dt)

Combined predict and update step.
"""
function step!(
    gpf::GaussianParticleFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    model_meas::AbstractMeasurementModel,
    u::AbstractVector,
    y::AbstractVector,
    dt::T = T(1.0)
) where {T, N}
    
    predict!(gpf, model_dyn, u, dt)
    update!(gpf, model_meas, y)
    return gpf
end

"""
    state_estimate(gpf::GaussianParticleFilter)

Return Gaussian mean estimate.
"""
function state_estimate(gpf::GaussianParticleFilter)
    return gpf.x̂
end

"""
    covariance(gpf::GaussianParticleFilter)

Return Gaussian covariance.
"""
function covariance(gpf::GaussianParticleFilter)
    return gpf.P
end

"""
    num_particles(gpf::GaussianParticleFilter)

Return number of particles.
"""
function num_particles(gpf::GaussianParticleFilter)
    return gpf.N_particles
end

"""
    effective_particles(gpf::GaussianParticleFilter)

Return effective sample size.
"""
function effective_particles(gpf::GaussianParticleFilter)
    return gpf.ess
end
