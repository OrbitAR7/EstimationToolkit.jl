"""
    GromovFlowFilter{T<:Real, N}

Gromov Flow Particle Filter - particle flow based on stochastic PDE.

Particle flow filters evolve particles continuously from prior to posterior
using a deterministic flow map, rather than importance sampling. This can
reduce particle degeneracy and avoid resampling.

# Mathematical Formulation

The Gromov flow evolves particles according to a stochastic PDE that
gradually incorporates measurement information:

```math
dx/dλ = -0.5 P(λ) Hᵀ R⁻¹ [y - h(x)] + √(P(λ)) dW/dλ
```

where λ ∈ [0,1] is the pseudo-time parameter that interpolates from
prior (λ=0) to posterior (λ=1).

# Algorithm
1. Start with particles from predicted distribution
2. Flow particles from λ=0 to λ=1 using SDE integration
3. Final particles represent posterior (no weights needed)

# Advantages
- No importance weights or resampling needed
- Can work better for highly informative measurements
- Maintains particle diversity

# Limitations
- Computationally expensive (SDE integration)
- Requires tunin of flow parameters
- Research-level method (less mature than SIS/SIR)

# Reference
- R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 3: Nonlinear Estimation (UT Austin ASE 381P.7, 2023), §6.1 (Gromov flow)
- Lecture18-PFF (Particle Flow Filters)
- Daum & Huang (2003, 2008)
- Reich (2013) - "A dynamical systems framework for intermittent data assimilation"

# Fields
- `particles::Vector{SVector{N,T}}` - Particle ensemble (no weights)
- `N_particles::Int` - Number of particles
- `n_flow_steps::Int` - Number of SDE integration steps
- `flow_dt::T` - Step size for flow integration
"""
mutable struct GromovFlowFilter{T<:Real, N} <: AbstractFilter
    particles::Vector{SVector{N, T}}
    N_particles::Int
    
    # Flow parameters
    n_flow_steps::Int
    flow_dt::T
    
    # RNG
    rng::AbstractRNG
    
    # Storage for mean/covariance
    x̂::SVector{N, T}
    P::SMatrix{N, N, T}
end

"""
    GromovFlowFilter(μ0, P0, N_particles; kwargs...)

Create Gromov Flow Filter.

# Arguments
- `μ0::SVector{N,T}` - Initial mean
- `P0::SMatrix{N,N,T}` - Initial covariance
- `N_particles::Int` - Number of particles

# Keyword Arguments
- `n_flow_steps::Int = 10` - Number of steps for flow integration
- `flow_dt::T = 0.1` - Step size for pseudo-time integration
- `rng::AbstractRNG = Random.GLOBAL_RNG`
"""
function GromovFlowFilter(
    μ0::SVector{N, T},
    P0::SMatrix{N, N, T},
    N_particles::Int;
    n_flow_steps::Int = 10,
    flow_dt::T = T(0.1),
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {T<:Real, N}
    
    @assert N_particles > 0 "Need at least one particle"
    @assert n_flow_steps > 0 "Need at least one flow step"
    
    # Sample initial particles
    L = cholesky(Symmetric(P0)).L
    particles = [μ0 + L * SVector{N, T}(randn(rng, N)) for _ in 1:N_particles]
    
    return GromovFlowFilter{T, N}(
        particles, N_particles, n_flow_steps, flow_dt, rng, μ0, P0
    )
end

"""
    predict!(gff::GromovFlowFilter, model_dyn, u, dt)

Prediction step: propagate particles through dynamics.
"""
function predict!(
    gff::GromovFlowFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    u::AbstractVector = zeros(T, 0),
    dt::T = T(1.0)
) where {T, N}
    
    # Propagate each particle
    for i in eachindex(gff.particles)
        v = sample_process_noise(model_dyn, dt, gff.rng)
        x_next = propagate(model_dyn, gff.particles[i], u, dt)
        gff.particles[i] = x_next + v
    end
    
    # Compute predicted mean and covariance
    gff.x̂ = mean(gff.particles)
    gff.P = cov_particles(gff.particles, gff.x̂)
    
    return gff
end

"""
    update!(gff::GromovFlowFilter, model_meas, y)

Measurement update: flow particles from prior to posterior.
"""
function update!(
    gff::GromovFlowFilter{T, N},
    model_meas::AbstractMeasurementModel,
    y::AbstractVector
) where {T, N}
    
    R = measurement_noise_cov(model_meas)
    R_inv = inv(R)
    
    # Flow each particle from λ=0 to λ=1
    dλ = T(1.0) / gff.n_flow_steps
    
    for step in 1:gff.n_flow_steps
        λ = step * dλ
        
        # Compute ensemble covariance at current flow step
        x̂_ens = mean(gff.particles)
        P_ens = cov_particles(gff.particles, x̂_ens)
        P_ens = ensure_spd(P_ens; ϵ=1e-6)
        
        # Cholesky for stochastic term
        L = try
            cholesky(Symmetric(P_ens)).L
        catch
            # Fallback to eigen-decomposition square root if Cholesky fails
            E = eigen(Symmetric(P_ens))
            # Ensure eigenvalues are non-negative
            λ = max.(E.values, zero(T))
            # Compute square root: L = V * sqrt(D)
            L_mat = E.vectors * Diagonal(sqrt.(λ))
            # Ensure we return a compatible matrix type
             if P_ens isa SMatrix
                SMatrix{N, N, T}(L_mat)
            else
                L_mat
            end
        end
        
        for i in eachindex(gff.particles)
            x = gff.particles[i]
            
            # Predicted measurement
            ŷ = measure(model_meas, x)
            
            # Measurement Jacobian (if nonlinear)
            H = jacobian_x(model_meas, x)
            
            # Innovation
            innov = y - ŷ
            
            # Deterministic drift: 0.5 P Hᵀ R⁻¹ [y - h(x)]
            drift = 0.5 * P_ens * H' * (R_inv * innov)
            
            # Stochastic term: √P dW
            dW = SVector{N, T}(randn(gff.rng, N) * sqrt(dλ))
            diffusion = L * dW
            
            # Update particle position
            gff.particles[i] = x + drift * dλ + diffusion
        end
    end
    
    # Compute posterior mean and covariance
    gff.x̂ = mean(gff.particles)
    gff.P = cov_particles(gff.particles, gff.x̂)
    
    return gff
end

"""
    step!(gff, model_dyn, model_meas, u, y, dt)

Combined predict and update step.
"""
function step!(
    gff::GromovFlowFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    model_meas::AbstractMeasurementModel,
    u::AbstractVector,
    y::AbstractVector,
    dt::T = T(1.0)
) where {T, N}
    
    predict!(gff, model_dyn, u, dt)
    update!(gff, model_meas, y)
    return gff
end

"""
    mean(particles::Vector{SVector{N,T}})

Compute ensemble mean.
"""
function mean(particles::Vector{SVector{N, T}}) where {N, T}
    return sum(particles) / length(particles)
end

"""
    cov_particles(particles, mean)

Compute ensemble covariance.
"""
function cov_particles(particles::Vector{SVector{N, T}}, x̂::SVector{N, T}) where {N, T}
    P = zeros(SMatrix{N, N, T})
    for x in particles
        diff = x - x̂
        P += diff * diff'
    end
    return make_symmetric(P / (length(particles) - 1))
end

"""
    state_estimate(gff::GromovFlowFilter)

Return ensemble mean.
"""
function state_estimate(gff::GromovFlowFilter)
    return gff.x̂
end

"""
    covariance(gff::GromovFlowFilter)

Return ensemble covariance.
"""
function covariance(gff::GromovFlowFilter)
    return gff.P
end

"""
    num_particles(gff::GromovFlowFilter)

Return number of particles.
"""
function num_particles(gff::GromovFlowFilter)
    return gff.N_particles
end


"""
    DaumHuangFlowFilter{T<:Real, N}

Daum-Huang Exact Particle Flow Filter (deterministic flow).

This is a deterministic version of particle flow that uses an exact
log-homotopy to flow particles from prior to posterior without
stochastic terms.

# Reference
- R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 3: Nonlinear Estimation (UT Austin ASE 381P.7, 2023), §6.2 (Daum-Huang exact flow)
- Daum & Huang (2009, 2010)
- "Exact particle flow for nonlinear filters"

# Note
This is a simplified placeholder implementation. Full implementation
requires solving the exact flow ODE which is more complex.
"""
mutable struct DaumHuangFlowFilter{T<:Real, N} <: AbstractFilter
    particles::Vector{SVector{N, T}}
    N_particles::Int
    n_flow_steps::Int
    rng::AbstractRNG
    x̂::SVector{N, T}
    P::SMatrix{N, N, T}
end

"""
    DaumHuangFlowFilter(μ0, P0, N_particles; kwargs...)

Create Daum-Huang Flow Filter (simplified).
"""
function DaumHuangFlowFilter(
    μ0::SVector{N, T},
    P0::SMatrix{N, N, T},
    N_particles::Int;
    n_flow_steps::Int = 20,
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {T<:Real, N}
    
    # Sample initial particles
    L = cholesky(Symmetric(P0)).L
    particles = [μ0 + L * SVector{N, T}(randn(rng, N)) for _ in 1:N_particles]
    
    return DaumHuangFlowFilter{T, N}(particles, N_particles, n_flow_steps, rng, μ0, P0)
end

"""
    predict!(dhf::DaumHuangFlowFilter, model_dyn, u, dt)

Prediction step.
"""
function predict!(
    dhf::DaumHuangFlowFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    u::AbstractVector = zeros(T, 0),
    dt::T = T(1.0)
) where {T, N}
    
    for i in eachindex(dhf.particles)
        v = sample_process_noise(model_dyn, dt, dhf.rng)
        dhf.particles[i] = propagate(model_dyn, dhf.particles[i], u, dt) + v
    end
    
    dhf.x̂ = mean(dhf.particles)
    dhf.P = cov_particles(dhf.particles, dhf.x̂)
    
    return dhf
end

"""
    update!(dhf::DaumHuangFlowFilter, model_meas, y)

Deterministic flow update (simplified gradient descent).
"""
function update!(
    dhf::DaumHuangFlowFilter{T, N},
    model_meas::AbstractMeasurementModel,
    y::AbstractVector
) where {T, N}
    
    R = measurement_noise_cov(model_meas)
    R_inv = inv(R)
    
    dλ = T(1.0) / dhf.n_flow_steps
    
    # Simplified deterministic flow (gradient-based)
    for step in 1:dhf.n_flow_steps
        for i in eachindex(dhf.particles)
            x = dhf.particles[i]
            ŷ = measure(model_meas, x)
            H = jacobian_x(model_meas, x)
            innov = y - ŷ
            
            # Gradient flow (simplified)
            grad = H' * (R_inv * innov)
            dhf.particles[i] = x + grad * dλ
        end
    end
    
    dhf.x̂ = mean(dhf.particles)
    dhf.P = cov_particles(dhf.particles, dhf.x̂)
    
    return dhf
end

"""
    step!(dhf, model_dyn, model_meas, u, y, dt)

Combined step.
"""
function step!(
    dhf::DaumHuangFlowFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    model_meas::AbstractMeasurementModel,
    u::AbstractVector,
    y::AbstractVector,
    dt::T = T(1.0)
) where {T, N}
    
    predict!(dhf, model_dyn, u, dt)
    update!(dhf, model_meas, y)
    return dhf
end

state_estimate(dhf::DaumHuangFlowFilter) = dhf.x̂
covariance(dhf::DaumHuangFlowFilter) = dhf.P
num_particles(dhf::DaumHuangFlowFilter) = dhf.N_particles
