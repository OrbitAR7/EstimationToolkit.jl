"""
    RegularizedParticleFilter

Regularized Particle Filter (RPF) using Kernel Density Estimation (KDE).

The RPF addresses particle degeneracy and sample impoverishment by using kernel
density estimation to create a continuous approximation of the particle distribution
before resampling. This "regularization" helps maintain particle diversity and
prevents loss of information in the tails of the distribution.

# Algorithm Overview

Standard resampling can lead to:
- **Particle degeneracy**: Few particles have significant weight
- **Sample impoverishment**: Particles collapse to few distinct values

Regularized resampling adds:
1. **KDE approximation**: p̂(x) = Σᵢ wᵢ·K(x - xᵢ, H) where K is kernel, H is bandwidth
2. **Continuous resampling**: Sample from KDE instead of discrete particles
3. **Bandwidth selection**: Optimal bandwidth H minimizes AMISE

# Mathematical Foundation

**Kernel Density Estimate**:
```
p̂(x) = Σᵢ wᵢ · (1/|H|^(1/2)) · K(H^(-1/2)·(x - xᵢ))
```

**Epanechnikov Kernel** (optimal MSE):
```
K(u) = (3/4)·(1 - u'u) if ||u|| < 1, else 0
```

**Optimal Bandwidth** (Silverman's rule of thumb):
```
H = A·Σ̂·n^(-2/(d+4))  where A = (4/(d+2))^(2/(d+4))
```

**Reference**: Musso, Oudjane, & Le Gland (2001), "Improving Regularized Particle Filters"
**Lecture Source**: 03-Nonlinear_filters.pdf §5.4

# Fields
- `particles::Vector{SVector{N,T}}` - Current particle states
- `weights::Vector{T}` - Normalized particle weights
- `resample_threshold::T` - ESS threshold for resampling
- `resample_method::Symbol` - Base resampling method
- `kernel::Symbol` - Kernel type (:epanechnikov or :gaussian)
- `bandwidth_scale::T` - Bandwidth scaling factor (default: 1.0)

# Constructor
```julia
RegularizedParticleFilter(x0::SVector{N,T}, P0::SMatrix{N,N,T}, N_particles::Int;
                          resample_threshold=N_particles/2,
                          resample_method=:systematic,
                          kernel=:epanechnikov,
                          bandwidth_scale=1.0)
```

# Example
```julia
# Initialize with 1000 particles
rpf = RegularizedParticleFilter(x0, P0, 1000)

# Filtering cycle
for (u, y) in zip(controls, measurements)
    predict!(rpf, dynamics_model, u, dt)
    update!(rpf, measurement_model, y)
end

# Better particle diversity than standard PF
ess = effective_particles(rpf)  # Should remain higher
```

# Performance Characteristics
- **Time complexity**: O(N²) for KDE evaluation (can be optimized)
- **Space complexity**: O(N·n) for N particles in n dimensions
- **Advantages**: Better particle diversity, smoother estimates
- **Disadvantages**: Higher computational cost, bandwidth tuning required
- **Best use**: When particle diversity is critical, moderate N (~100-1000)
"""
mutable struct RegularizedParticleFilter{T<:Real, N} <: AbstractFilter
    particles::Vector{SVector{N, T}}
    weights::Vector{T}
    resample_threshold::T
    resample_method::Symbol
    kernel::Symbol
    bandwidth_scale::T
    
    function RegularizedParticleFilter(x0::SVector{N,T}, P0::SMatrix{N,N,T}, N_particles::Int;
                                      resample_threshold=N_particles/2,
                                      resample_method=:systematic,
                                      kernel::Symbol=:epanechnikov,
                                      bandwidth_scale::T=T(1.0)) where {N,T}
        # Sample particles from initial distribution
        P0_sym = make_symmetric(P0)
        P0_spd = ensure_spd(P0_sym)
        dist = MvNormal(Vector(x0), Matrix(P0_spd))
        
        particles = [SVector{N,T}(rand(dist)) for _ in 1:N_particles]
        weights = fill(T(1.0 / N_particles), N_particles)
        
        new{T,N}(particles, weights, T(resample_threshold), resample_method, 
                 kernel, bandwidth_scale)
    end
end

"""
    compute_optimal_bandwidth(particles, weights)

Compute optimal bandwidth matrix using Silverman's rule of thumb.

# Formula
H = A·Σ̂·n^(-2/(d+4))

where:
- A = (4/(d+2))^(2/(d+4)) is dimension-dependent constant
- Σ̂ is sample covariance
- n is effective sample size
- d is dimension
"""
function compute_optimal_bandwidth(particles::Vector{SVector{N,T}}, weights::Vector{T}) where {N,T}
    # Compute weighted covariance
    μ = sample_mean(particles, weights)
    Σ = sample_covariance(particles, weights, μ)
    
    # Effective sample size
    n_eff = effective_sample_size(weights)
    
    # Silverman's constant
    d = N
    A = (4 / (d + 2))^(2 / (d + 4))
    
    # Optimal bandwidth: H = A·Σ·n^(-2/(d+4))
    scale = A * n_eff^(-2 / (d + 4))
    H = scale * Σ
    
    # Ensure SPD
    H_sym = make_symmetric(H)
    return ensure_spd(H_sym)
end

"""
    sample_from_kernel(particle, bandwidth, kernel_type)

Sample from kernel centered at particle with given bandwidth matrix.

# Kernels
- :gaussian - Standard Gaussian kernel (unbounded support)
- :epanechnikov - Epanechnikov kernel (compact support, MSE-optimal)
"""
function sample_from_kernel(particle::SVector{N,T}, bandwidth::SMatrix{N,N,T}, 
                           kernel_type::Symbol) where {N,T}
    if kernel_type == :gaussian
        # Gaussian kernel: K(u) = (2π)^(-d/2) exp(-0.5 u'u)
        # Sample: x = particle + √H · z where z ~ N(0,I)
        H_sqrt = SMatrix{N,N,T}(cholesky(Symmetric(Matrix(bandwidth))).L)
        z = SVector{N,T}(randn(T, N))
        return particle + H_sqrt * z
        
    elseif kernel_type == :epanechnikov
        # Epanechnikov kernel: K(u) = (3/4)(1 - u'u) for ||u|| < 1
        # Sample from truncated distribution
        # Approximation: sample from uniform sphere and rescale
        
        # Sample from multivariate uniform ball
        # Method: sample from Gaussian, normalize, scale by U^(1/d)
        z = randn(T, N)
        z_norm = z / norm(z)  # Unit sphere
        u = rand(T)^(1/N)  # Radial component
        
        # Transform to bandwidth space
        H_sqrt = SMatrix{N,N,T}(cholesky(Symmetric(Matrix(bandwidth))).L)
        displacement = u * SVector{N,T}(z_norm)
        
        return particle + H_sqrt * displacement
    else
        error("Unknown kernel type: $kernel_type")
    end
end

"""
    regularized_resample!(rpf::RegularizedParticleFilter)

Perform regularized resampling using KDE.

# Algorithm
1. Compute optimal bandwidth H
2. Standard resampling to get indices
3. Add kernel noise to each resampled particle
4. Reset weights to uniform
"""
function regularized_resample!(rpf::RegularizedParticleFilter{T,N}) where {T,N}
    # Compute optimal bandwidth
    H = compute_optimal_bandwidth(rpf.particles, rpf.weights)
    H_scaled = rpf.bandwidth_scale * H
    
    # Standard resampling to get indices
    if rpf.resample_method == :systematic
        indices = systematic_resample(rpf.weights)
    elseif rpf.resample_method == :stratified
        indices = stratified_resample(rpf.weights)
    else
        indices = multinomial_resample(rpf.weights)
    end
    
    # Resample and add kernel noise (regularization)
    N_particles = length(rpf.particles)
    new_particles = Vector{SVector{N,T}}(undef, N_particles)
    
    for i in 1:N_particles
        # Get resampled particle
        base_particle = rpf.particles[indices[i]]
        
        # Add kernel noise for regularization
        new_particles[i] = sample_from_kernel(base_particle, H_scaled, rpf.kernel)
    end
    
    # Update particles and reset weights
    rpf.particles = new_particles
    fill!(rpf.weights, 1.0 / N_particles)
end

"""
    predict!(rpf::RegularizedParticleFilter, model::AbstractDynamicsModel, u, dt)

Regularized particle filter prediction step.
"""
function predict!(rpf::RegularizedParticleFilter{T,N}, model::AbstractDynamicsModel, u, dt) where {T,N}
    # Propagate particles through dynamics with process noise
    @inbounds for i in eachindex(rpf.particles)
        process_noise = sample_process_noise(model, dt)
        rpf.particles[i] = propagate(model, rpf.particles[i], u, dt) + SVector{N,T}(process_noise)
    end
end

"""
    update!(rpf::RegularizedParticleFilter, model::AbstractMeasurementModel, y)

Regularized particle filter measurement update with KDE-based resampling.
"""
function update!(rpf::RegularizedParticleFilter{T,N}, model::AbstractMeasurementModel, y::SVector) where {T,N}
    N_particles = length(rpf.particles)
    
    # Get measurement noise covariance
    R = measurement_noise_cov(model)
    R_sym = make_symmetric(R)
    R_inv = inv(R_sym)
    
    # Update weights based on measurement likelihood
    @inbounds for i in 1:N_particles
        # Predicted measurement
        y_pred = measure(model, rpf.particles[i])
        
        # Innovation
        ν = y - y_pred
        
        # Log-likelihood (Gaussian): log p(y|x) = -0.5 * ν' * R⁻¹ * ν - log normalization
        log_likelihood = -0.5 * dot(ν, R_inv * ν)
        
        # Update weight
        rpf.weights[i] *= exp(log_likelihood)
    end
    
    # Normalize weights
    sum_weights = sum(rpf.weights)
    if sum_weights > 0
        rpf.weights ./= sum_weights
    else
        # Fallback to uniform weights
        fill!(rpf.weights, 1.0 / N_particles)
    end
    
    # Check if regularized resampling is needed
    ess = effective_sample_size(rpf.weights)
    
    if ess < rpf.resample_threshold
        regularized_resample!(rpf)
    end
end

"""
    state_estimate(rpf::RegularizedParticleFilter)

Compute weighted mean of particle distribution.
"""
function state_estimate(rpf::RegularizedParticleFilter)
    return sample_mean(rpf.particles, rpf.weights)
end

"""
    covariance(rpf::RegularizedParticleFilter)

Compute weighted covariance of particle distribution.
"""
function covariance(rpf::RegularizedParticleFilter)
    μ = state_estimate(rpf)
    return sample_covariance(rpf.particles, rpf.weights, μ)
end

"""
    num_particles(rpf::RegularizedParticleFilter)

Return the number of particles.
"""
function num_particles(rpf::RegularizedParticleFilter)
    return length(rpf.particles)
end

"""
    effective_particles(rpf::RegularizedParticleFilter)

Return the effective sample size.
"""
function effective_particles(rpf::RegularizedParticleFilter)
    return effective_sample_size(rpf.weights)
end
