# Noise modeling utilities

"""
    sample_process_noise(model::AbstractDynamicsModel, dt, rng=Random.GLOBAL_RNG)

Sample from the process noise distribution.

Default implementation assumes zero-mean Gaussian.
"""
function sample_process_noise(model::AbstractDynamicsModel{NX, NU, NV}, dt, 
                               rng=Random.GLOBAL_RNG) where {NX, NU, NV}
    Q = process_noise_cov(model, dt)
    # Use Cholesky for sampling: v ~ N(0, Q) => v = L*z where z ~ N(0, I)
    L = cholesky(Symmetric(Q)).L
    z = randn(rng, NV)
    return L * z
end

"""
    sample_measurement_noise(model::AbstractMeasurementModel, rng=Random.GLOBAL_RNG)

Sample from the measurement noise distribution.

Default implementation assumes zero-mean Gaussian.
"""
function sample_measurement_noise(model::AbstractMeasurementModel{NX, NY, NW}, 
                                   rng=Random.GLOBAL_RNG) where {NX, NY, NW}
    R = measurement_noise_cov(model)
    L = cholesky(Symmetric(R)).L
    z = randn(rng, NW)
    return L * z
end

"""
    simulate_trajectory(dyn_model, x0, controls, dt, T; rng=Random.GLOBAL_RNG)

Simulate a trajectory using the dynamics model with process noise.

# Arguments
- `dyn_model`: Dynamics model
- `x0`: Initial state
- `controls`: Vector of control inputs (or `nothing` for each step)
- `dt`: Time step
- `T`: Number of time steps

Returns a vector of states.
"""
function simulate_trajectory(dyn_model, x0, controls, dt, T; rng=Random.GLOBAL_RNG)
    states = Vector{typeof(x0)}(undef, T+1)
    states[1] = x0
    
    for k in 1:T
        u = controls[k]
        v = sample_process_noise(dyn_model, dt, rng)
        
        # Propagate with noise
        x_next = propagate(dyn_model, states[k], u, dt)
        # Add process noise (assuming additive for now)
        states[k+1] = x_next + v
    end
    
    return states
end

"""
    simulate_measurements(meas_model, states; rng=Random.GLOBAL_RNG)

Generate noisy measurements from a trajectory.

Returns a vector of measurements.
"""
function simulate_measurements(meas_model, states; rng=Random.GLOBAL_RNG)
    measurements = []
    
    for x in states
        y_clean = measure(meas_model, x)
        w = sample_measurement_noise(meas_model, rng)
        y_noisy = y_clean + w
        push!(measurements, y_noisy)
    end
    
    return measurements
end
