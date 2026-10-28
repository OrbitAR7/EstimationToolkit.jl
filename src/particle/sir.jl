"""
    SequentialImportanceResampling{T<:Real, N}

General Sequential Importance Resampling (SIR) particle filter framework.

The SIR filter is a flexible particle filtering framework that allows for
arbitrary proposal distributions. The bootstrap filter is a special case where
the proposal is the transition density.

# General Algorithm
1. **Prediction/Proposal**: Sample x_k^(i) ~ q(x_k | x_{k-1}^(i), y_k)
2. **Importance Weighting**: w_k^(i) ∝ w_{k-1}^(i) · p(y_k | x_k^(i)) · p(x_k^(i) | x_{k-1}^(i)) / q(x_k^(i) | x_{k-1}^(i), y_k)
3. **Normalization**: w_k^(i) = w_k^(i) / Σⱼ w_k^(j)
4. **Resampling** (adaptive): If ESS < threshold, resample particles

# Proposal Distributions
The proposal function should have signature:
    `proposal(x_prev, u, y, model_dyn, model_meas, dt, rng) -> x_new`

Common proposals:
- **Bootstrap (Transition)**: q = p(x_k | x_{k-1})  [default]
- **Optimal**: q = p(x_k | x_{k-1}, y_k)  [requires analytical/numerical integration]
- **EKF-based**: Use EKF to approximate optimal proposal
- **UKF-based**: Use UKF to approximate optimal proposal

# Fields
- `particles::Vector{SVector{N, T}}`: Particle states
- `weights::Vector{T}`: Normalized particle weights
- `proposal::Function`: Proposal distribution function
- `resample_method::Symbol`: Resampling algorithm
- `resample_threshold::T`: ESS threshold for resampling
- `rng::AbstractRNG`: Random number generator

# References
- R. Zanetti, "Sequential Importance Sampling," ASE 381P.7 lecture slides (UT Austin, 2023)
- R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 3: Nonlinear Estimation (UT Austin ASE 381P.7, 2023), Ch. 5
- Arulampalam et al. (2002). "A Tutorial on Particle Filters"
- Doucet & Johansen (2009). "A Tutorial on Particle Filtering and Smoothing"

# Notes
For the importance weight computation, we need to evaluate:
- Likelihood: p(y_k | x_k^(i))  [provided by measurement model]
- Transition: p(x_k^(i) | x_{k-1}^(i))  [provided by dynamics model]
- Proposal: q(x_k^(i) | x_{k-1}^(i), y_k)  [user-provided or default]

The bootstrap proposal simplifies the weight update to just the likelihood.
"""
mutable struct SequentialImportanceResampling{T<:Real, N} <: AbstractFilter
    particles::Vector{SVector{N, T}}
    weights::Vector{T}
    
    # Proposal distribution (function)
    proposal::Function
    
    # Resampling configuration
    resample_method::Symbol
    resample_threshold::T
    
    # RNG for reproducibility
    rng::AbstractRNG
    
    # Store last measurement for diagnostics
    last_measurement::Union{Nothing, SVector}
    last_predicted_measurement::Union{Nothing, SVector}
    
    function SequentialImportanceResampling(particles::Vector{SVector{N, T}},
                                           weights::Vector{T},
                                           proposal::Function;
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
        
        new{T, N}(particles, weights, proposal, resample_method, 
                 resample_threshold, rng, nothing, nothing)
    end
end

# Convenience constructor with initial distribution
function SequentialImportanceResampling(μ0::AbstractVector{T}, P0::AbstractMatrix{T}, 
                                       N_particles::Int, proposal::Function;
                                       kwargs...) where {T<:Real}
    n = length(μ0)
    
    # Sample initial particles from Gaussian
    L = cholesky(Symmetric(P0)).L
    particles = [SVector{n, T}(μ0 + L * randn(n)) for _ in 1:N_particles]
    weights = fill(1.0 / N_particles, N_particles)
    
    return SequentialImportanceResampling(particles, weights, proposal; kwargs...)
end

"""
    bootstrap_proposal(x_prev, u, y, model_dyn, model_meas, dt, rng)

Bootstrap proposal: sample from transition density p(x | x_prev).

This is the simplest proposal and doesn't use the measurement y.
The importance weight simplifies to just the likelihood p(y | x).
"""
function bootstrap_proposal(x_prev::SVector{N, T}, u, y, model_dyn, model_meas, 
                           dt, rng) where {N, T}
    # Propagate through dynamics
    x_pred = propagate(model_dyn, x_prev, u, dt)
    
    # Add process noise
    v = sample_process_noise(model_dyn, dt, rng)
    
    return x_pred + SVector{N, T}(v)
end

"""
    predict_and_propose!(sir::SequentialImportanceResampling, model_dyn, model_meas, u, y, dt)

Combined prediction and proposal step for SIR filter.

For general proposals that may depend on the measurement, we need to:
1. Sample new particles from proposal: x_k^(i) ~ q(x_k | x_{k-1}^(i), y_k)
2. Update weights according to importance sampling ratio

For bootstrap proposal, this simplifies since the proposal equals the transition.
"""
function predict_and_propose!(sir::SequentialImportanceResampling{T, N}, 
                              model_dyn::AbstractDynamicsModel,
                              model_meas::AbstractMeasurementModel,
                              u, y, dt) where {T, N}
    # Sample from proposal for each particle
    new_particles = Vector{SVector{N, T}}(undef, length(sir.particles))
    
    for i in eachindex(sir.particles)
        new_particles[i] = sir.proposal(sir.particles[i], u, y, model_dyn, 
                                       model_meas, dt, sir.rng)
    end
    
    sir.particles = new_particles
    
    return sir
end

"""
    update_weights!(sir::SequentialImportanceResampling, model_dyn, model_meas, u, y, dt)

Update importance weights according to:
    w_k^(i) ∝ w_{k-1}^(i) · [p(y_k | x_k^(i)) · p(x_k^(i) | x_{k-1}^(i))] / q(x_k^(i) | ...)

For bootstrap proposal where q = p(x_k | x_{k-1}), this simplifies to:
    w_k^(i) ∝ w_{k-1}^(i) · p(y_k | x_k^(i))

# Note
This function assumes the proposal has already been applied (particles updated).
For bootstrap, we only need to compute the measurement likelihood.
"""
function update_weights!(sir::SequentialImportanceResampling{T, N},
                        model_dyn::AbstractDynamicsModel,
                        model_meas::AbstractMeasurementModel,
                        u, y, dt) where {T, N}
    # Compute likelihood for each particle
    for i in eachindex(sir.particles)
        # Predicted measurement
        ŷ = measure(model_meas, sir.particles[i])
        
        # Compute likelihood: p(y | x^(i))
        # Assuming Gaussian measurement noise: p(y | x) = N(y; h(x), R)
        R = measurement_noise_cov(model_meas)
        residual = y - ŷ
        
        # Multivariate Gaussian likelihood (unnormalized is sufficient)
        exponent = -0.5 * (residual' * (R \ residual))
        likelihood = exp(exponent)
        
        # Update weight (for bootstrap, this is just the likelihood)
        sir.weights[i] *= likelihood
    end
    
    # Normalize weights
    w_sum = sum(sir.weights)
    if w_sum > 0
        sir.weights ./= w_sum
    else
        # All weights zero - reset to uniform (degenerate case)
        sir.weights .= 1.0 / length(sir.weights)
    end
    
    # Store diagnostics
    sir.last_measurement = y
    
    return sir
end

"""
    adaptive_resample!(sir::SequentialImportanceResampling)

Adaptively resample particles based on effective sample size (ESS).

Resamples only when ESS/N < threshold to avoid unnecessary resampling.
"""
function adaptive_resample!(sir::SequentialImportanceResampling)
    # Compute effective sample size
    ess = effective_sample_size(sir.weights)
    n_particles = length(sir.particles)
    
    # Resample if needed
    if ess / n_particles < sir.resample_threshold
        indices = if sir.resample_method == :systematic
            systematic_resample(sir.weights, sir.rng)
        elseif sir.resample_method == :stratified
            stratified_resample(sir.weights, sir.rng)
        else  # :multinomial
            multinomial_resample(sir.weights, sir.rng)
        end
        
        # Resample particles
        sir.particles = [sir.particles[i] for i in indices]
        
        # Reset weights to uniform
        sir.weights = fill(1.0 / n_particles, n_particles)
    end
    
    return sir
end

"""
    step!(sir::SequentialImportanceResampling, dyn, meas, u, y, dt)

Complete SIR filter step: predict, propose, update weights, and resample.

This is the main interface for using the SIR filter.
"""
function step!(sir::SequentialImportanceResampling, 
               dyn::AbstractDynamicsModel,
               meas::AbstractMeasurementModel, 
               u, y, dt)
    # 1. Prediction and proposal
    predict_and_propose!(sir, dyn, meas, u, y, dt)
    
    # 2. Importance weight update
    update_weights!(sir, dyn, meas, u, y, dt)
    
    # 3. Adaptive resampling
    adaptive_resample!(sir)
    
    return sir
end

"""
    state_estimate(sir::SequentialImportanceResampling)

Return weighted mean of particles as state estimate.
"""
function state_estimate(sir::SequentialImportanceResampling{T, N}) where {T, N}
    return sample_mean(sir.particles, sir.weights)
end

"""
    covariance(sir::SequentialImportanceResampling)

Return weighted sample covariance as error covariance estimate.
"""
function covariance(sir::SequentialImportanceResampling{T, N}) where {T, N}
    μ = state_estimate(sir)
    Σ = sample_covariance(sir.particles, sir.weights, μ)
    return SMatrix{N, N, T}(Σ)
end

"""
    state_covariance(sir::SequentialImportanceResampling)

Alias for covariance() for consistency with particle belief interface.
"""
function state_covariance(sir::SequentialImportanceResampling)
    return covariance(sir)
end

"""
    effective_particles(sir::SequentialImportanceResampling)

Return the effective number of particles.
"""
function effective_particles(sir::SequentialImportanceResampling)
    return effective_sample_size(sir.weights)
end

"""
    num_particles(sir::SequentialImportanceResampling)

Return the total number of particles.
"""
function num_particles(sir::SequentialImportanceResampling)
    return length(sir.particles)
end
