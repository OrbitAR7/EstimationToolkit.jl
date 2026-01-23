"""
Orbit Determination Example

Demonstrates estimation of satellite state from range and range-rate measurements
using various filters. This is a classic aerospace application showing the strength
of different filtering approaches on a real problem.

State: x = [r, ṙ, θ, θ̇]'
- r: radial distance (km)
- ṙ: radial velocity (km/s)
- θ: angular position (rad)
- θ̇: angular velocity (rad/s)

Dynamics: Simplified Keplerian motion with perturbations
Measurements: Range and range-rate from ground station
"""

using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random
using Statistics
using Distributions
# using Plots  # Uncomment to enable plotting (requires Plots.jl package)
const PLOTTING_ENABLED = false

# Orbital dynamics model (simplified)
struct OrbitalDynamics{T} <: AbstractDynamicsModel{4, 0, 4}
    μ::T  # Gravitational parameter (km³/s²)
    q::T  # Process noise intensity
end

function EstimationToolkit.propagate(model::OrbitalDynamics{T}, x, u, dt) where {T}
    r, ṙ, θ, θ̇ = x[1], x[2], x[3], x[4]
    
    # Simplified orbital equations (linearized for this example)
    # In reality, would use full nonlinear propagator
    r̈ = r * θ̇^2 - model.μ / r^2
    θ̈ = -(2 * ṙ * θ̇) / r
    
    # Forward Euler for demonstration (real code would use RK4 or similar)
    r_new = r + ṙ * dt
    ṙ_new = ṙ + r̈ * dt
    θ_new = θ + θ̇ * dt
    θ̇_new = θ̇ + θ̈ * dt
    
    return @SVector [r_new, ṙ_new, θ_new, θ̇_new]
end

function EstimationToolkit.process_noise_cov(model::OrbitalDynamics{T}, dt::T) where {T}
    q = model.q
    return @SMatrix [
        q*dt^3/3  q*dt^2/2  0         0;
        q*dt^2/2  q*dt      0         0;
        0         0         q*dt^3/3  q*dt^2/2;
        0         0         q*dt^2/2  q*dt
    ]
end

# Range and range-rate measurements from ground station
struct GroundStationMeasurement{T} <: AbstractMeasurementModel{4, 2, 2}
    station_pos::SVector{2, T}  # [r_station, θ_station]
    σ_range::T
    σ_range_rate::T
end

function EstimationToolkit.measure(model::GroundStationMeasurement{T}, x) where {T}
    r, ṙ, θ, θ̇ = x[1], x[2], x[3], x[4]
    r_s, θ_s = model.station_pos[1], model.station_pos[2]
    
    # Range: distance from satellite to ground station
    Δθ = θ - θ_s
    range = sqrt(r^2 + r_s^2 - 2*r*r_s*cos(Δθ))
    
    # Range-rate: derivative of range
    range_rate = (r*ṙ - r_s*r*θ̇*sin(Δθ)) / range
    
    return @SVector [range, range_rate]
end

function EstimationToolkit.measurement_noise_cov(model::GroundStationMeasurement{T}) where {T}
    return @SMatrix [model.σ_range^2  0;
                     0                model.σ_range_rate^2]
end

"""
    run_orbit_determination(; duration=3600, dt=30.0, plot_results=true)

Run orbit determination example comparing multiple filters.

# Arguments
- `duration`: Simulation duration (seconds)
- `dt`: Time step (seconds)
- `plot_results`: Whether to generate plots
"""
function run_orbit_determination(; duration=600.0, dt=1.0, plot_results=PLOTTING_ENABLED)
    Random.seed!(42)
    
    # Earth gravitational parameter
    μ_earth = 398600.4418  # km³/s²
    
    # True initial state (LEO satellite, ~400 km altitude)
    r0 = 6378.0 + 400.0  # km
    v0 = sqrt(μ_earth / r0)  # Circular orbit velocity
    x_true = @SVector [r0, 0.0, 0.0, v0/r0]
    
    # Dynamics and measurement models
    dynamics = OrbitalDynamics(μ_earth, 1e-8)
    ground_station = GroundStationMeasurement(@SVector([6378.0, 0.0]), 0.1, 0.01)
    
    # Initial estimate (with error)
    x0 = x_true + @SVector [10.0, 0.1, 0.001, 0.00001]
    P0 = @SMatrix [100.0  0     0       0;
                   0      1.0   0       0;
                   0      0     0.01    0;
                   0      0     0       1e-6]
    
    # Initialize filters
    kf = KalmanFilter(x0, P0)  # Will be EKF actually
    ekf = ExtendedKalmanFilter(x0, P0)
    ukf = UnscentedKalmanFilter(x0, P0, α=1e-3, β=2.0, κ=0.0)
    
    # Simulate and estimate
    n_steps = Int(duration / dt)
    time = 0:dt:duration
    
    x_true_hist = Vector{SVector{4, Float64}}(undef, n_steps+1)
    measurements = Vector{SVector{2, Float64}}(undef, n_steps+1)
    x_kf = Vector{SVector{4, Float64}}(undef, n_steps+1)
    x_ekf = Vector{SVector{4, Float64}}(undef, n_steps+1)
    x_ukf = Vector{SVector{4, Float64}}(undef, n_steps+1)
    
    x_true_hist[1] = x_true
    x_kf[1] = state_estimate(kf)
    x_ekf[1] = state_estimate(ekf)
    x_ukf[1] = state_estimate(ukf)
    
    x_current = x_true
    
    println("Running orbit determination simulation...")
    println("Filters: KF, EKF, UKF")
    println("Duration: $(duration/60) minutes, dt: $(dt) seconds\n")
    
    for k in 1:n_steps
        # Propagate true state
        Q = process_noise_cov(dynamics, dt)
        v = MvNormal(zeros(4), Matrix(Q))
        x_current = propagate(dynamics, x_current, zeros(0), dt) + rand(v)
        x_true_hist[k+1] = x_current
        
        # Generate measurement
        R = measurement_noise_cov(ground_station)
        w = MvNormal(zeros(2), Matrix(R))
        z = measure(ground_station, x_current) + rand(w)
        measurements[k+1] = z
        
        # Run filters
        predict!(kf, dynamics, nothing, dt)
        predict!(ekf, dynamics, nothing, dt)
        predict!(ukf, dynamics, nothing, dt)
        
        update!(kf, ground_station, z)
        update!(ekf, ground_station, z)
        update!(ukf, ground_station, z)
        
        x_kf[k+1] = state_estimate(kf)
        x_ekf[k+1] = state_estimate(ekf)
        x_ukf[k+1] = state_estimate(ukf)
    end
    
    # Compute errors
    errors_kf = [norm(x_true_hist[k] - x_kf[k]) for k in 1:n_steps+1]
    errors_ekf = [norm(x_true_hist[k] - x_ekf[k]) for k in 1:n_steps+1]
    errors_ukf = [norm(x_true_hist[k] - x_ukf[k]) for k in 1:n_steps+1]
    
    println("Final RMSE (km):")
    println("  KF:  $(sqrt(mean(errors_kf.^2)))")
    println("  EKF: $(sqrt(mean(errors_ekf.^2)))")
    println("  UKF: $(sqrt(mean(errors_ukf.^2)))")
    
    results = (
        time = time,
        truth = x_true_hist,
        measurements = measurements,
        kf = x_kf,
        ekf = x_ekf,
        ukf = x_ukf,
        errors = (kf=errors_kf, ekf=errors_ekf, ukf=errors_ukf)
    )
    
    if plot_results
        plot_orbit_results(results)
    end
    
    return results
end

function plot_orbit_results(results)
    t = collect(results.time) ./ 60  # Convert to minutes
    
    # Plot 1: Radial position error
    p1 = plot(t, [x[1] - results.truth[i][1] for (i,x) in enumerate(results.kf)],
              label="KF", linewidth=2, xlabel="Time (min)", ylabel="Radial Error (km)",
              title="Radial Position Error", legend=:topright)
    plot!(p1, t, [x[1] - results.truth[i][1] for (i,x) in enumerate(results.ekf)],
          label="EKF", linewidth=2)
    plot!(p1, t, [x[1] - results.truth[i][1] for (i,x) in enumerate(results.ukf)],
          label="UKF", linewidth=2)
    
    # Plot 2: Position errors over time
    p2 = plot(t, results.errors.kf, label="KF", linewidth=2,
              xlabel="Time (min)", ylabel="Position Error (km)",
              title="Total Position Error", legend=:topright)
    plot!(p2, t, results.errors.ekf, label="EKF", linewidth=2)
    plot!(p2, t, results.errors.ukf, label="UKF", linewidth=2)
    
    # Plot 3: Orbit trajectory (polar)
    r_true = [x[1] for x in results.truth]
    θ_true = [x[3] for x in results.truth]
    r_ukf = [x[1] for x in results.ukf]
    θ_ukf = [x[3] for x in results.ukf]
    
    x_true = r_true .* cos.(θ_true)
    y_true = r_true .* sin.(θ_true)
    x_ukf = r_ukf .* cos.(θ_ukf)
    y_ukf = r_ukf .* sin.(θ_ukf)
    
    p3 = plot(x_true, y_true, label="True", linewidth=2, aspect_ratio=:equal,
              xlabel="X (km)", ylabel="Y (km)", title="Orbital Trajectory")
    plot!(p3, x_ukf, y_ukf, label="UKF Estimate", linewidth=2, linestyle=:dash)
    scatter!(p3, [0], [0], label="Earth", markersize=10, color=:blue)
    
    # Combine plots
    plot(p1, p2, p3, layout=(1,3), size=(1400, 400))
    savefig("orbit_determination_results.png")
    println("\nPlot saved as orbit_determination_results.png")
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_orbit_determination()
end
