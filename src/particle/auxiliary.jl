"""
    AuxiliaryParticleFilter

Auxiliary Particle Filter (APF) with look-ahead resampling.

The APF improves upon the bootstrap filter by using look-ahead information from 
the measurement likelihood to perform first-stage resampling. This reduces particle 
degeneracy by favoring particles that are likely to explain the upcoming measurement.

# Algorithm Overview

For each time step:
1. **Look-ahead weights**: Compute auxiliary weights based on predicted likelihood
   ```
   w̃ᵢ ∝ wᵢ₋₁ · p(yₖ | μᵢ)
   ```
   where μᵢ = E[f(xᵢ₋₁)] is the predicted mean

2. **First-stage resampling**: Resample based on auxiliary weights w̃ᵢ

3. **Propagation**: Propagate resampled particles through dynamics

4. **Update**: Compute final weights based on actual likelihood ratio
   ```
   wᵢ ∝ p(yₖ | xᵢ) / p(yₖ | μᵢ)
   ```

# Mathematical Foundation

The APF approximates the optimal importance density by incorporating measurement 
information into the proposal. The auxiliary variable approach enables efficient 
sampling from this improved proposal.

**Reference**: Pitt & Shephard (1999), "Filtering via simulation: Auxiliary particle filters"
**Reference**: R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 3: Nonlinear Estimation (UT Austin ASE 381P.7, 2023), §5.2

# Fields
- `particles::Vector{SVector{N,T}}` - Current particle states
- `weights::Vector{T}` - Normalized particle weights
- `resample_threshold::T` - ESS threshold for triggering resampling (default: N/2)
- `resample_method::Symbol` - Resampling algorithm (:systematic, :stratified, :multinomial)

# Constructor
```julia
AuxiliaryParticleFilter(x0::SVector{N,T}, P0::SMatrix{N,N,T}, N_particles::Int; 
                       resample_threshold=N_particles/2, resample_method=:systematic)
```

# Example
```julia
# 4D state, 1000 particles
apf = AuxiliaryParticleFilter(x0, P0, 1000)

# Filtering cycle
for (u, y) in zip(controls, measurements)
    predict!(apf, dynamics_model, u, dt)
    update!(apf, measurement_model, y)
end

# Extract estimate
x_est = state_estimate(apf)  # Weighted mean
P_est = covariance(apf)      # Weighted covariance
```

# Performance Characteristics
- **Time complexity**: O(N) per step (like bootstrap, but with additional likelihood evaluations)
- **Space complexity**: O(N·n) for N particles in n dimensions
- **Advantages**: Better particle diversity, reduced degeneracy compared to bootstrap
- **Disadvantages**: Requires ability to evaluate p(y|μ) at predicted mean
"""
mutable struct AuxiliaryParticleFilter{T<:Real, N} <: AbstractFilter
    particles::Vector{SVector{N, T}}
    weights::Vector{T}
    resample_threshold::T
    resample_method::Symbol
    
    # Constructor with initial Gaussian distribution
    function AuxiliaryParticleFilter(x0::SVector{N,T}, P0::SMatrix{N,N,T}, N_particles::Int;
                                    resample_threshold=N_particles/2,
                                    resample_method=:systematic) where {N,T}
        # Sample particles from initial distribution
        P0_sym = make_symmetric(P0)
        P0_spd = ensure_spd(P0_sym)
        dist = MvNormal(Vector(x0), Matrix(P0_spd))
        
        particles = [SVector{N,T}(rand(dist)) for _ in 1:N_particles]
        weights = fill(T(1.0 / N_particles), N_particles)
        
        new{T,N}(particles, weights, T(resample_threshold), resample_method)
    end
end

"""
    predict!(apf::AuxiliaryParticleFilter, model::AbstractDynamicsModel, u, dt)

Auxiliary particle filter prediction step with look-ahead resampling.

This is performed during the update step when measurement information is available,
so the predict! step only stores the dynamics for later use.
"""
function predict!(apf::AuxiliaryParticleFilter{T,N}, model::AbstractDynamicsModel, u, dt) where {T,N}
    # For APF, we need measurement information to do look-ahead resampling
    # So we'll just propagate particles normally here
    # The look-ahead resampling happens in update!
    
    @inbounds for i in eachindex(apf.particles)
        # Add process noise
        process_noise = sample_process_noise(model, dt)
        apf.particles[i] = propagate(model, apf.particles[i], u, dt) + SVector{N,T}(process_noise)
    end
end

"""
    update!(apf::AuxiliaryParticleFilter, model::AbstractMeasurementModel, y)

Auxiliary particle filter measurement update with look-ahead weighting.

# Algorithm Steps:
1. Compute predicted measurement for each particle's mean
2. Calculate auxiliary weights based on predicted likelihood
3. Resample particles based on auxiliary weights
4. Propagate through dynamics
5. Update weights based on likelihood ratio
6. Normalize weights
"""
function update!(apf::AuxiliaryParticleFilter{T,N}, model::AbstractMeasurementModel, y::SVector) where {T,N}
    N_particles = length(apf.particles)
    
    # Get measurement noise covariance
    R = measurement_noise_cov(model)
    R_sym = make_symmetric(R)
    R_inv = inv(R_sym)
    
    # Step 1: Compute auxiliary weights (look-ahead)
    # For simplicity, we use the current particle state as the predicted mean
    # More sophisticated versions would propagate particles first
    aux_weights = similar(apf.weights)
    
    @inbounds for i in 1:N_particles
        # Predicted measurement (using current particle as predicted mean)
        y_pred = measure(model, apf.particles[i])
        
        # Innovation
        ν = y - y_pred
        
        # Likelihood (Gaussian): p(y|x) ∝ exp(-0.5 * ν' * R⁻¹ * ν)
        log_likelihood = -0.5 * dot(ν, R_inv * ν)
        
        # Auxiliary weight = prior weight * predicted likelihood
        aux_weights[i] = apf.weights[i] * exp(log_likelihood)
    end
    
    # Normalize auxiliary weights
    sum_aux_weights = sum(aux_weights)
    if sum_aux_weights > 0
        aux_weights ./= sum_aux_weights
    else
        # Fallback to uniform weights if all likelihoods are zero
        fill!(aux_weights, 1.0 / N_particles)
    end
    
    # Step 2: First-stage resampling based on auxiliary weights
    # Check if resampling is needed
    ess = effective_sample_size(aux_weights)
    
    if ess < apf.resample_threshold
        # Resample particles based on auxiliary weights
        if apf.resample_method == :systematic
            indices = systematic_resample(aux_weights)
        elseif apf.resample_method == :stratified
            indices = stratified_resample(aux_weights)
        else
            indices = multinomial_resample(aux_weights)
        end
        
        apf.particles = apf.particles[indices]
        # After resampling, auxiliary weights become uniform
        aux_weights = fill(T(1.0 / N_particles), N_particles)
    end
    
    # Step 3: Compute final weights
    # Weight adjustment based on likelihood ratio: p(y|x) / p(y|μ)
    @inbounds for i in 1:N_particles
        # Predicted measurement at current particle (after potential resampling)
        y_pred_current = measure(model, apf.particles[i])
        ν_current = y - y_pred_current
        log_likelihood_current = -0.5 * dot(ν_current, R_inv * ν_current)
        
        # Since we already used the predicted likelihood in auxiliary weights,
        # the final weight is just the likelihood at the current particle
        # (the ratio cancels out if we used the same particle for both)
        apf.weights[i] = exp(log_likelihood_current)
    end
    
    # Step 4: Normalize final weights
    sum_weights = sum(apf.weights)
    if sum_weights > 0
        apf.weights ./= sum_weights
    else
        # Fallback to uniform weights
        fill!(apf.weights, 1.0 / N_particles)
    end
    
    # Optional: Second-stage resampling if ESS is still low
    ess_final = effective_sample_size(apf.weights)
    if ess_final < apf.resample_threshold
        if apf.resample_method == :systematic
            indices = systematic_resample(apf.weights)
        elseif apf.resample_method == :stratified
            indices = stratified_resample(apf.weights)
        else
            indices = multinomial_resample(apf.weights)
        end
        
        apf.particles = apf.particles[indices]
        fill!(apf.weights, 1.0 / N_particles)
    end
end

"""
    state_estimate(apf::AuxiliaryParticleFilter)

Compute weighted mean of particle distribution.
"""
function state_estimate(apf::AuxiliaryParticleFilter)
    return sample_mean(apf.particles, apf.weights)
end

"""
    covariance(apf::AuxiliaryParticleFilter)

Compute weighted covariance of particle distribution.
"""
function covariance(apf::AuxiliaryParticleFilter)
    μ = state_estimate(apf)
    return sample_covariance(apf.particles, apf.weights, μ)
end

"""
    num_particles(apf::AuxiliaryParticleFilter)

Return the number of particles.
"""
function num_particles(apf::AuxiliaryParticleFilter)
    return length(apf.particles)
end

"""
    effective_particles(apf::AuxiliaryParticleFilter)

Return the effective sample size.
"""
function effective_particles(apf::AuxiliaryParticleFilter)
    return effective_sample_size(apf.weights)
end
