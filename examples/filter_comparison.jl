"""
Filter Comparison Study

Systematic comparison of different filter types across varying:
- Nonlinearity levels
- Noise conditions
- State dimensions

Helps users understand when to use each filter type.
"""

using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random
using Statistics
using Printf
using Distributions

# Test dynamics with tunable nonlinearity
struct TestDynamics{T} <: AbstractDynamicsModel{2, 0, 2}
    nonlinearity::T  # 0 = linear, 1 = moderate, 2 = strong
    q::T
end

function EstimationToolkit.propagate(model::TestDynamics{T}, x, u, dt) where {T}
    α = model.nonlinearity
    # x_{k+1} = x_k + dt * f(x_k) where f(x) = x + α*sin(x)
    f1 = x[1] + α * sin(x[1])
    f2 = x[2] + α * sin(x[2])
    return x + dt * @SVector [f1, f2]
end

EstimationToolkit.process_noise_cov(m::TestDynamics{T}, dt::T) where {T} =
    m.q * dt * @SMatrix [1.0 0.0; 0.0 1.0]

# Test measurement with tunable nonlinearity
struct TestMeasurement{T} <: AbstractMeasurementModel{2, 2, 2}
    nonlinearity::T
    r::T
end

function EstimationToolkit.measure(model::TestMeasurement{T}, x) where {T}
    α = model.nonlinearity
    if α < 0.1
        # Linear measurement
        return x
    else
        # Nonlinear measurement
        return @SVector [x[1] + α*x[1]^2, x[2] + α*x[2]^2]
    end
end

EstimationToolkit.measurement_noise_cov(m::TestMeasurement{T}) where {T} =
    m.r^2 * @SMatrix [1.0 0.0; 0.0 1.0]

"""
    compare_filters(; nonlinearity=0.5, process_noise=0.1, meas_noise=0.5, n_steps=100)

Run comparison study across multiple filter types.
"""
function compare_filters(;
    nonlinearity::Float64 = 0.5,
    process_noise::Float64 = 0.1,
    meas_noise::Float64 = 0.5,
    n_steps::Int = 100,
    n_monte_carlo::Int = 50
)
    println("\n" * "="^70)
    println("Filter Comparison Study")
    println("="^70)
    println("Nonlinearity: $(nonlinearity)")
    println("Process noise: $(process_noise)")
    println("Measurement noise: $(meas_noise)")
    println("Monte Carlo runs: $(n_monte_carlo)")
    println("="^70 * "\n")
    
    # Storage for results
    rmse_results = Dict{String, Vector{Float64}}()
    time_results = Dict{String, Float64}()
    
    for mc in 1:n_monte_carlo
        Random.seed!(mc)
        
        # True initial state
        x_true = @SVector [1.0, 1.0]
        
        # Perturbed initial estimate
        x0 = x_true + @SVector [0.5, 0.5]
        P0 = @SMatrix [1.0 0.0; 0.0 1.0]
        
        # Models
        dynamics = TestDynamics(nonlinearity, process_noise)
        measurement = TestMeasurement(nonlinearity * 0.5, meas_noise)
        
        # Initialize filters
        kf = KalmanFilter(x0, P0)
        ekf = ExtendedKalmanFilter(x0, P0)
        iekf = IteratedExtendedKalmanFilter(x0, P0, max_iterations=3)
        ukf = UnscentedKalmanFilter(x0, P0)
        ckf = CubatureKalmanFilter(x0, P0)
        
        # Particle filter (smaller for speed)
        pf = BootstrapParticleFilter(fill(x0, 200), fill(1.0/200, 200))
        
        filters = [
            ("KF", kf),
            ("EKF", ekf),
            ("IEKF", iekf),
            ("UKF", ukf),
            ("CKF", ckf),
            ("PF", pf)
        ]
        
        # Simulate truth
        x_current = x_true
        truth = Vector{SVector{2, Float64}}(undef, n_steps+1)
        measurements = Vector{SVector{2, Float64}}(undef, n_steps+1)
        truth[1] = x_true
        
        for k in 1:n_steps
            Q = process_noise_cov(dynamics, 0.1)
            x_current = propagate(dynamics, x_current, zeros(0), 0.1) + 
                        rand(MvNormal(zeros(2), Matrix(Q)))
            truth[k+1] = x_current
            
            R = measurement_noise_cov(measurement)
            measurements[k+1] = measure(measurement, x_current) +
                               rand(MvNormal(zeros(2), Matrix(R)))
        end
        
        # Run each filter
        for (name, filter) in filters
            errors = Float64[]
            
            start_time = time()
            for k in 1:n_steps
                predict!(filter, dynamics, nothing, 0.1)
                update!(filter, measurement, measurements[k+1])
                
                x_est = state_estimate(filter)
                error = norm(x_est - truth[k+1])
                push!(errors, error)
            end
            elapsed = time() - start_time
            
            # Store results
            if !haskey(rmse_results, name)
                rmse_results[name] = Float64[]
                time_results[name] = 0.0
            end
            push!(rmse_results[name], sqrt(mean(errors.^2)))
            time_results[name] += elapsed
        end
    end
    
    # Compute statistics
    println("\nResults (averaged over $n_monte_carlo runs):")
    println("-"^70)
    @printf("%-10s | %12s | %12s | %12s\n", "Filter", "Mean RMSE", "Std RMSE", "Time (ms)")
    println("-"^70)
    
    filter_order = ["KF", "EKF", "IEKF", "UKF", "CKF", "PF"]
    for name in filter_order
        if !haskey(rmse_results, name); continue; end
        
        mean_rmse = mean(rmse_results[name])
        std_rmse = std(rmse_results[name])
        avg_time = (time_results[name] / n_monte_carlo) * 1000  # to ms
        
        @printf("%-10s | %12.4f | %12.4f | %12.2f\n", 
                name, mean_rmse, std_rmse, avg_time)
    end
    println("-"^70)
    
    # Best filter
    best_filter = argmin(name -> mean(rmse_results[name]), keys(rmse_results))
    println("\nBest performer (lowest RMSE): $best_filter")
    
    # Recommendations
    println("\nRecommendations:")
    if nonlinearity < 0.2
        println("  - For this linear/weakly nonlinear case, KF or EKF are sufficient")
        println("  - UKF/CKF add overhead without benefit")
    elseif nonlinearity < 0.8
        println("  - Moderate nonlinearity: UKF or CKF recommended")
        println("  - IEKF may improve over EKF with iteration")
    else
        println("  - Strong nonlinearity: Consider UKF, CKF, or particle filter")
        println("  - Particle filter more robust but computationally expensive")
    end
    
    return (rmse=rmse_results, time=time_results)
end

"""
    run_nonlinearity_sweep()

Compare filters across different nonlinearity levels.
"""
function run_nonlinearity_sweep()
    println("\n" * "="^70)
    println("Nonlinearity Sweep Study")
    println("="^70 * "\n")
    
    nonlinearities = [0.0, 0.2, 0.5, 1.0, 1.5]
    filters = ["KF", "EKF", "UKF", "CKF"]
    
    results = Dict(filter => Float64[] for filter in filters)
    
    for α in nonlinearities
        println("Testing nonlinearity level: $α")
        comparison = compare_filters(
            nonlinearity=α,
            process_noise=0.1,
            meas_noise=0.5,
            n_steps=50,
            n_monte_carlo=20
        )
        
        for filter in filters
            push!(results[filter], mean(comparison.rmse[filter]))
        end
    end
    
    # Print summary table
    println("\n" * "="^70)
    println("Summary: RMSE vs Nonlinearity Level")
    println("="^70)
    @printf("%-15s |", "Nonlinearity")
    for filter in filters
        @printf(" %10s |", filter)
    end
    println()
    println("-"^70)
    
    for (i, α) in enumerate(nonlinearities)
        @printf("%-15.2f |", α)
        for filter in filters
            @printf(" %10.4f |", results[filter][i])
        end
        println()
    end
    println("="^70)
    
    return results
end

"""
    run_noise_study()

Compare filter performance across different noise levels.
"""
function run_noise_study()
    println("\n" * "="^70)
    println("Noise Level Study")
    println("="^70 * "\n")
    
    noise_levels = [0.01, 0.1, 0.5, 1.0, 2.0]
    filters = ["EKF", "UKF", "CKF", "PF"]
    
    results = Dict(filter => Float64[] for filter in filters)
    
    for σ in noise_levels
        println("Testing measurement noise: $σ")
        comparison = compare_filters(
            nonlinearity=0.5,
            process_noise=0.1,
            meas_noise=σ,
            n_steps=50,
            n_monte_carlo=20
        )
        
        for filter in filters
            push!(results[filter], mean(comparison.rmse[filter]))
        end
    end
    
    # Print summary
    println("\n" * "="^70)
    println("Summary: RMSE vs Measurement Noise Level")
    println("="^70)
    @printf("%-15s |", "Noise σ")
    for filter in filters
        @printf(" %10s |", filter)
    end
    println()
    println("-"^70)
    
    for (i, σ) in enumerate(noise_levels)
        @printf("%-15.2f |", σ)
        for filter in filters
            @printf(" %10.4f |", results[filter][i])
        end
        println()
    end
    println("="^70)
    
    return results
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running filter comparison studies...\n")
    
    # Single comparison
    println("1. Single scenario comparison:")
    compare_filters(nonlinearity=0.5, n_monte_carlo=30)
    
    # Nonlinearity sweep
    println("\n\n2. Nonlinearity sweep:")
    run_nonlinearity_sweep()
    
    # Noise study
    println("\n\n3. Noise sensitivity study:")
    run_noise_study()
    
    println("\n\nAll studies complete!")
end
