"""
Constant Velocity Tracking Example

Simple 2D tracking problem with linear dynamics but potentially nonlinear measurements.

State: x = [px, py, vx, vy]'
- px, py: position
- vx, vy: velocity

Dynamics: Constant velocity model with process noise
Measurement: Position only (can extend to range-bearing)
"""

using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random
using Statistics

# Define constant velocity dynamics model
struct ConstantVelocityModel{T} <: AbstractDynamicsModel{4, 0, 4}
    q::T  # Process noise intensity
end

function EstimationToolkit.propagate(model::ConstantVelocityModel{T}, x, u, dt) where {T}
    # State transition matrix for constant velocity
    F = @SMatrix [
        1.0 0.0  dt 0.0;
        0.0 1.0 0.0  dt;
        0.0 0.0 1.0 0.0;
        0.0 0.0 0.0 1.0
    ]
    return F * x
end

function EstimationToolkit.jacobian_x(model::ConstantVelocityModel, x::SVector, u, dt)
    # Jacobian is just F for this linear model
    return @SMatrix [
        1.0 0.0  dt 0.0;
        0.0 1.0 0.0  dt;
        0.0 0.0 1.0 0.0;
        0.0 0.0 0.0 1.0
    ]
end

function EstimationToolkit.process_noise_cov(model::ConstantVelocityModel{T}, dt) where {T}
    # Continuous white noise acceleration model
    # Q = G * q * G' where G is the noise gain matrix
    q = model.q
    
    q11 = dt^3/3 * q
    q13 = dt^2/2 * q
    q33 = dt * q
    
    Q = SMatrix{4,4,T}(
        q11, 0.0, q13, 0.0,
        0.0, q11, 0.0, q13,
        q13, 0.0, q33, 0.0,
        0.0, q13, 0.0, q33
    )
    
    return Q
end

# Define position measurement model
struct PositionMeasurementModel{T} <: AbstractMeasurementModel{4, 2, 2}
    R::SMatrix{2, 2, T}  # Measurement noise covariance
end

function EstimationToolkit.measure(model::PositionMeasurementModel, x)
    # Measure only position
    H = @SMatrix [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]
    return H * x
end

function EstimationToolkit.jacobian_x(model::PositionMeasurementModel, x::SVector)
    return @SMatrix [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]
end

EstimationToolkit.measurement_noise_cov(model::PositionMeasurementModel) = model.R

# Simulation parameters
function run_tracking_example(; use_kf=true, use_ekf=true, use_ukf=true, use_pf=true)
    Random.seed!(42)
    
    dt = 0.1  # Time step (s)
    T = 100   # Number of steps
    
    # True initial state
    x_true_0 = @SVector [0.0, 0.0, 1.0, 0.5]
    
    # Models
    dyn_model = ConstantVelocityModel(0.1)
    meas_model = PositionMeasurementModel(@SMatrix [0.1 0.0; 0.0 0.1])
    
    # Generate true trajectory and measurements
    println("Generating trajectory...")
    x_true = [x_true_0]
    for k in 1:T
        x_next = propagate(dyn_model, x_true[k], nothing, dt)
        v = sample_process_noise(dyn_model, dt)
        push!(x_true, x_next + v)
    end
    
    measurements = [measure(meas_model, x) + sample_measurement_noise(meas_model) 
                   for x in x_true]
    
    # Initial estimate (with uncertainty)
    x0_est = x_true_0 + @SVector [0.5, 0.5, 0.0, 0.0]
    P0 = @SMatrix [1.0 0.0 0.0 0.0;
                   0.0 1.0 0.0 0.0;
                   0.0 0.0 0.5 0.0;
                   0.0 0.0 0.0 0.5]
    
    results = Dict()
    
    # Kalman Filter
    if use_kf
        println("\nRunning Kalman Filter...")
        kf = KalmanFilter(x0_est, P0)
        kf_estimates = [kf.x]
        kf_nis = Float64[]
        
        for k in 1:T
            predict!(kf, dyn_model, nothing, dt)
            update!(kf, meas_model, measurements[k+1])
            push!(kf_estimates, kf.x)
            push!(kf_nis, nis(kf))
        end
        
        kf_rmse = sqrt(mean([sum((xt - xe).^2) for (xt, xe) in zip(x_true, kf_estimates)]))
        println("  RMSE: $(round(kf_rmse, digits=4))")
        println("  Mean NIS: $(round(mean(kf_nis), digits=2)) (expected: 2.0)")
        
        results[:kf] = (estimates=kf_estimates, nis=kf_nis, rmse=kf_rmse)
    end
    
    # Extended Kalman Filter (same as KF for linear system, but tests the implementation)
    if use_ekf
        println("\nRunning Extended Kalman Filter...")
        ekf = ExtendedKalmanFilter(x0_est, P0)
        ekf_estimates = [ekf.x]
        
        for k in 1:T
            predict!(ekf, dyn_model, nothing, dt)
            update!(ekf, meas_model, measurements[k+1])
            push!(ekf_estimates, ekf.x)
        end
        
        ekf_rmse = sqrt(mean([sum((xt - xe).^2) for (xt, xe) in zip(x_true, ekf_estimates)]))
        println("  RMSE: $(round(ekf_rmse, digits=4))")
        
        results[:ekf] = (estimates=ekf_estimates, rmse=ekf_rmse)
    end
    
    # Unscented Kalman Filter
    if use_ukf
        println("\nRunning Unscented Kalman Filter...")
        ukf = UnscentedKalmanFilter(x0_est, P0)
        ukf_estimates = [ukf.x]
        
        for k in 1:T
            predict!(ukf, dyn_model, nothing, dt)
            update!(ukf, meas_model, measurements[k+1])
            push!(ukf_estimates, ukf.x)
        end
        
        ukf_rmse = sqrt(mean([sum((xt - xe).^2) for (xt, xe) in zip(x_true, ukf_estimates)]))
        println("  RMSE: $(round(ukf_rmse, digits=4))")
        
        results[:ukf] = (estimates=ukf_estimates, rmse=ukf_rmse)
    end
    
    # Bootstrap Particle Filter
    if use_pf
        println("\nRunning Bootstrap Particle Filter...")
        N_particles = 500
        pf = BootstrapParticleFilter(x0_est, P0, N_particles)
        pf_estimates = [state_estimate(pf)]
        
        for k in 1:T
            predict!(pf, dyn_model, nothing, dt)
            update!(pf, meas_model, measurements[k+1])
            push!(pf_estimates, state_estimate(pf))
            
            if k % 20 == 0
                ess = effective_particles(pf)
                println("  Step $k: ESS = $(round(ess, digits=1)) / $N_particles")
            end
        end
        
        pf_rmse = sqrt(mean([sum((xt - xe).^2) for (xt, xe) in zip(x_true, pf_estimates)]))
        println("  RMSE: $(round(pf_rmse, digits=4))")
        
        results[:pf] = (estimates=pf_estimates, rmse=pf_rmse)
    end
    
    println("\n" * "="^50)
    println("Tracking Example Complete!")
    println("="^50)
    
    return results, x_true, measurements
end

# Run the example if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results, x_true, measurements = run_tracking_example()
end
