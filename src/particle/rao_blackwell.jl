"""
    RaoBlackwellizedParticleFilter{T<:Real, NL, NN}

Rao-Blackwellized Particle Filter (RBPF) / Marginalized Particle Filter.

The RBPF exploits conditionally linear Gaussian substructure in the state space.
When the state can be partitioned into linear and nonlinear components, the linear
part can be estimated optimally using Kalman filters, while particles handle only
the nonlinear part. This dramatically reduces the required number of particles.

# State Structure

Partition state: x = [xₗ, xₙ]
- xₗ: Linear substate (conditionally Gaussian given xₙ)
- xₙ: Nonlinear substate

# Mathematical Formulation

Dynamics:
```math
xₗ(k+1) = Aₖ(xₙ) xₗ(k) + Bₖ(xₙ) u + vₗ(k)
xₙ(k+1) = fₙ(xₙ(k), xₗ(k)) + vₙ(k)
```

Measurement:
```math
y(k) = H(xₙ) xₗ(k) + h(xₙ) + w(k)
```

Algorithm:
1. Sample nonlinear particles: xₙ⁽ⁱ⁾
2. For each particle, run Kalman filter for xₗ⁽ⁱ⁾ | xₙ⁽ⁱ⁾
3. Update weights based on marginal likelihood p(y | xₙ⁽ⁱ⁾)
4. Resample particles

# Advantages
- Reduces particle degeneracy (fewer dimensions for particles)
- Optimal estimation of linear substate
- Can achieve same performance with orders of magnitude fewer particles

# Example Applications
- SLAM (robot pose nonlinear, map linear)
- Target tracking with uncertain parameters
- Orbit determination with linear dynamics

# Reference
- 03-Nonlinear_filters.pdf §5.8 (Rao-Blackwellization)
- Doucet et al. (2000)
- Schön et al. (2005)

# Fields
- `particles_nonlinear::Vector{SVector{NN,T}}` - Nonlinear state particles
- `kalman_filters::Vector{KF}` - Kalman filters for linear substate (one per particle)
- `weights::Vector{T}` - Particle weights
"""
mutable struct RaoBlackwellizedParticleFilter{T<:Real, NL, NN, KF<:KalmanFilter} <: AbstractFilter
    # Nonlinear particles
    particles_nonlinear::Vector{SVector{NN, T}}
    
    # Kalman filters for linear substate (one per particle)
    kalman_filters::Vector{KF}
    
    # Particle weights
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
    RaoBlackwellizedParticleFilter(μ0_linear, P0_linear, particles_nonlinear; kwargs...)

Create RBPF with initial conditions.

# Arguments
- `μ0_linear::SVector{NL,T}` - Initial mean for linear substate
- `P0_linear::SMatrix{NL,NL,T}` - Initial covariance for linear substate
- `particles_nonlinear::Vector{SVector{NN,T}}` - Initial nonlinear particles
- `resample_threshold::T = 0.5` - ESS threshold for resampling
- `resample_method::Symbol = :systematic` - Resampling method
"""
function RaoBlackwellizedParticleFilter(
    μ0_linear::SVector{NL, T},
    P0_linear::SMatrix{NL, NL, T},
    particles_nonlinear::Vector{SVector{NN, T}};
    resample_threshold::T = T(0.5),
    resample_method::Symbol = :systematic,
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {T<:Real, NL, NN}
    
    N_particles = length(particles_nonlinear)
    @assert N_particles > 0 "Need at least one particle"
    
    # Initialize Kalman filter for each particle
    kalman_filters = [KalmanFilter(μ0_linear, P0_linear) for _ in 1:N_particles]
    
    weights = fill(T(1.0) / N_particles, N_particles)
    ess = T(N_particles)
    
    return RaoBlackwellizedParticleFilter{T, NL, NN, KalmanFilter{T, NL}}(
        particles_nonlinear, kalman_filters, weights, N_particles,
        resample_threshold, resample_method, rng, ess
    )
end

"""
    predict!(rbpf, model_dyn, u, dt)

RBPF prediction step.

# Arguments
- `model_dyn` - Dynamics model with methods:
  - `propagate_nonlinear(xₙ, xₗ, u, dt)` - Nonlinear state dynamics
  - `linear_dynamics_matrix(xₙ, u, dt)` - A matrix for linear substate
  - `linear_control_matrix(xₙ, u, dt)` - B matrix (optional)
  - `process_noise_cov_linear(xₙ, dt)` - Q for linear substate
  - `process_noise_cov_nonlinear(xₙ, dt)` - Q for nonlinear substate
"""
function predict!(
    rbpf::RaoBlackwellizedParticleFilter{T, NL, NN, KF},
    model_dyn,
    u::AbstractVector = zeros(T, 0),
    dt::T = T(1.0)
) where {T, NL, NN, KF}
    
    # For each particle
    for i in 1:rbpf.N_particles
        xₙ = rbpf.particles_nonlinear[i]
        
        # 1. Propagate nonlinear particle
        xₗ_mean = state_estimate(rbpf.kalman_filters[i])
        xₙ_new = propagate_nonlinear(model_dyn, xₙ, xₗ_mean, u, dt)
        
        # Add process noise to nonlinear state
        Qₙ = process_noise_cov_nonlinear(model_dyn, xₙ, dt)
        vₙ = SVector{NN, T}(rand(rbpf.rng, MvNormal(zeros(NN), Qₙ)))
        rbpf.particles_nonlinear[i] = xₙ_new + vₙ
        
        # 2. Propagate linear substate using KF prediction
        # Get linearized dynamics: xₗ(k+1) = A(xₙ) xₗ(k) + B(xₙ) u + vₗ
        A = linear_dynamics_matrix(model_dyn, xₙ, u, dt)
        
        # Manual KF prediction for conditionally linear system
        kf = rbpf.kalman_filters[i]
        x̂ = kf.x
        P = kf.P
        
        # Predict
        x̂_new = A * x̂
        if !isempty(u)
            B = linear_control_matrix(model_dyn, xₙ, u, dt)
            x̂_new += B * u
        end
        
        Qₗ = process_noise_cov_linear(model_dyn, xₙ, dt)
        P_new = A * P * A' + Qₗ
        P_new = make_symmetric(P_new)
        
        # Update Kalman filter
        kf.x = x̂_new
        kf.P = P_new
    end
    
    return rbpf
end

"""
    update!(rbpf, model_meas, y)

RBPF measurement update.

# Arguments
- `model_meas` - Measurement model with methods:
  - `linear_measurement_matrix(xₙ)` - H matrix for linear substate
  - `nonlinear_measurement(xₙ)` - h(xₙ) component
  - `measurement_noise_cov(xₙ)` - R matrix
"""
function update!(
    rbpf::RaoBlackwellizedParticleFilter{T, NL, NN, KF},
    model_meas,
    y::AbstractVector
) where {T, NL, NN, KF}
    
    # For each particle
    for i in 1:rbpf.N_particles
        xₙ = rbpf.particles_nonlinear[i]
        kf = rbpf.kalman_filters[i]
        
        # Get measurement model components
        H = linear_measurement_matrix(model_meas, xₙ)
        h = nonlinear_measurement(model_meas, xₙ)
        R = measurement_noise_cov(model_meas, xₙ)
        
        # Predicted measurement: ŷ = H xₗ + h(xₙ)
        ŷ = H * kf.x + h
        
        # Innovation
        ν = y - ŷ
        
        # Innovation covariance
        S = H * kf.P * H' + R
        S = make_symmetric(S)
        
        # Update weight based on marginal likelihood p(y | xₙ⁽ⁱ⁾, y₁:ₖ₋₁)
        # This is Gaussian: N(y; ŷ, S)
        log_like = -0.5 * (dot(ν, S \ ν) + logdet(S) + length(y) * log(2π))
        rbpf.weights[i] *= exp(log_like)
        
        # Kalman update for linear substate
        K = kf.P * H' / S
        kf.x = kf.x + K * ν
        kf.P = (I - K * H) * kf.P
        kf.P = make_symmetric(kf.P)
    end
    
    # Normalize weights
    total_weight = sum(rbpf.weights)
    if total_weight > eps(T)
        rbpf.weights ./= total_weight
    else
        rbpf.weights .= T(1.0) / rbpf.N_particles
    end
    
    rbpf.ess = effective_sample_size(rbpf.weights)
    
    # Resample if needed
    if rbpf.ess < rbpf.resample_threshold * rbpf.N_particles
        resample!(rbpf)
    end
    
    return rbpf
end

"""
    resample!(rbpf::RaoBlackwellizedParticleFilter)

Resample particles and their associated Kalman filters.
"""
function resample!(rbpf::RaoBlackwellizedParticleFilter{T, NL, NN, KF}) where {T, NL, NN, KF}
    # Resample indices
    if rbpf.resample_method == :systematic
        indices = systematic_resample(rbpf.weights; rng=rbpf.rng)
    elseif rbpf.resample_method == :stratified
        indices = stratified_resample(rbpf.weights; rng=rbpf.rng)
    else
        indices = multinomial_resample(rbpf.weights; rng=rbpf.rng)
    end
    
    # Resample particles
    rbpf.particles_nonlinear = rbpf.particles_nonlinear[indices]
    
    # Deep copy Kalman filters (important!)
    new_filters = Vector{KF}(undef, rbpf.N_particles)
    for i in 1:rbpf.N_particles
        old_kf = rbpf.kalman_filters[indices[i]]
        new_filters[i] = KalmanFilter(old_kf.x, old_kf.P)
    end
    rbpf.kalman_filters = new_filters
    
    # Reset weights
    rbpf.weights .= T(1.0) / rbpf.N_particles
    rbpf.ess = T(rbpf.N_particles)
    
    return rbpf
end

"""
    step!(rbpf, model_dyn, model_meas, u, y, dt)

Combined predict and update step.
"""
function step!(
    rbpf::RaoBlackwellizedParticleFilter{T, NL, NN, KF},
    model_dyn,
    model_meas,
    u::AbstractVector,
    y::AbstractVector,
    dt::T = T(1.0)
) where {T, NL, NN, KF}
    
    predict!(rbpf, model_dyn, u, dt)
    update!(rbpf, model_meas, y)
    return rbpf
end

"""
    state_estimate(rbpf::RaoBlackwellizedParticleFilter)

Compute MMSE estimate of full state [xₗ, xₙ].
"""
function state_estimate(rbpf::RaoBlackwellizedParticleFilter{T, NL, NN, KF}) where {T, NL, NN, KF}
    # Weighted average of nonlinear state
    x̂ₙ = zeros(SVector{NN, T})
    for i in 1:rbpf.N_particles
        x̂ₙ += rbpf.weights[i] * rbpf.particles_nonlinear[i]
    end
    
    # Weighted average of linear state estimates
    x̂ₗ = zeros(SVector{NL, T})
    for i in 1:rbpf.N_particles
        x̂ₗ += rbpf.weights[i] * state_estimate(rbpf.kalman_filters[i])
    end
    
    # Concatenate
    return vcat(x̂ₗ, x̂ₙ)
end

"""
    covariance(rbpf::RaoBlackwellizedParticleFilter)

Compute covariance of full state (approximate).
"""
function covariance(rbpf::RaoBlackwellizedParticleFilter{T, NL, NN, KF}) where {T, NL, NN, KF}
    x̂ = state_estimate(rbpf)
    x̂ₗ = x̂[1:NL]
    x̂ₙ = x̂[(NL+1):end]
    
    # Covariance blocks
    P_full = zeros(SMatrix{NL+NN, NL+NN, T})
    
    for i in 1:rbpf.N_particles
        xₗ_i = state_estimate(rbpf.kalman_filters[i])
        xₙ_i = rbpf.particles_nonlinear[i]
        x_i = vcat(xₗ_i, xₙ_i)
        
        diff = x_i - x̂
        P_full += rbpf.weights[i] * (diff * diff')
        
        # Add linear substate covariance
        Pₗ_i = covariance(rbpf.kalman_filters[i])
        
        # Construct block update [P_l 0; 0 0]
        # Using hcat/vcat avoids setindex issues with SMatrix
        top_right = zeros(SMatrix{NL, NN, T})
        bottom = zeros(SMatrix{NN, NL + NN, T})
        P_block = vcat(hcat(Pₗ_i, top_right), bottom)
        
        P_full += rbpf.weights[i] * P_block
    end
    
    return make_symmetric(P_full)
end

"""
    num_particles(rbpf::RaoBlackwellizedParticleFilter)

Return number of particles.
"""
function num_particles(rbpf::RaoBlackwellizedParticleFilter)
    return rbpf.N_particles
end

"""
    effective_particles(rbpf::RaoBlackwellizedParticleFilter)

Return effective sample size.
"""
function effective_particles(rbpf::RaoBlackwellizedParticleFilter)
    return rbpf.ess
end
