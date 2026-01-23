"""
EstimationToolkit.jl Benchmarks

Simple performance benchmarks for comparing filter implementations.
For detailed benchmarking, consider using BenchmarkTools.jl.

Run with:
    julia --project=. benchmark/benchmarks.jl
"""

using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random
using Printf

# Simple dynamics for benchmarking
struct BenchmarkDynamics{T} <: AbstractDynamicsModel{4, 0, 4}
    q::T
end

EstimationToolkit.propagate(m::BenchmarkDynamics{T}, x, u, dt) where {T} =
    @SMatrix([1.0 0.0 dt 0.0; 0.0 1.0 0.0 dt; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]) * x

EstimationToolkit.process_noise_cov(m::BenchmarkDynamics{T}, dt::T) where {T} =
    m.q * @SMatrix [dt^3/3 0 dt^2/2 0; 0 dt^3/3 0 dt^2/2; dt^2/2 0 dt 0; 0 dt^2/2 0 dt]

struct BenchmarkMeasurement{T} <: AbstractMeasurementModel{4, 2, 2}
    r::T
end

EstimationToolkit.measure(m::BenchmarkMeasurement{T}, x) where {T} = x[1:2]
EstimationToolkit.measurement_noise_cov(m::BenchmarkMeasurement{T}) where {T} =
    m.r^2 * @SMatrix [1.0 0.0; 0.0 1.0]

"""
    benchmark_filter(filter, dynamics, measurement, n_steps)

Benchmark a single filter over n_steps.
"""
function benchmark_filter(filter, dynamics, measurement, n_steps::Int)
    # Warmup
    for _ in 1:10
        predict!(filter, dynamics, nothing, 0.1)
        update!(filter, measurement, @SVector [0.0, 0.0])
    end
    
    # Actual benchmark
    start = time()
    for _ in 1:n_steps
        predict!(filter, dynamics, nothing, 0.1)
        update!(filter, measurement, @SVector [0.0, 0.0])
    end
    elapsed = time() - start
    
    return elapsed / n_steps * 1000  # ms per step
end

"""
    run_benchmarks(; n_steps=1000, verbose=true)

Run comprehensive benchmark suite.
"""
function run_benchmarks(; n_steps=1000, verbose=true)
    Random.seed!(42)
    
    x0 = @SVector [0.0, 0.0, 1.0, 0.5]
    P0 = SMatrix{4,4}(1.0I)
    
    dynamics = BenchmarkDynamics(0.1)
    measurement = BenchmarkMeasurement(0.5)
    
    filters = [
        ("KalmanFilter", KalmanFilter(x0, P0)),
        ("InformationFilter", InformationFilter(inv(P0), inv(P0) * x0)),
        ("SquareRootCovarianceFilter", SquareRootCovarianceFilter(x0, cholesky(P0).L)),
        ("SquareRootInformationFilter", SquareRootInformationFilter(cholesky(inv(P0)).U, cholesky(inv(P0)).U * x0)),
        ("ExtendedKalmanFilter", ExtendedKalmanFilter(x0, P0)),
        ("IteratedEKF", IteratedExtendedKalmanFilter(x0, P0, max_iterations=3)),
        ("SecondOrderEKF", SecondOrderExtendedKalmanFilter(x0, P0)),
        ("UnscentedKalmanFilter", UnscentedKalmanFilter(x0, P0)),
        ("CubatureKalmanFilter", CubatureKalmanFilter(x0, P0)),
        ("QuadratureKalmanFilter", QuadratureKalmanFilter(x0, P0, order=3)),
        ("DividedDifferenceFilter", DividedDifferenceFilter(x0, P0)),
        ("BootstrapPF (100)", BootstrapParticleFilter(fill(x0, 100), fill(0.01, 100))),
        ("BootstrapPF (500)", BootstrapParticleFilter(fill(x0, 500), fill(0.002, 500))),
    ]
    
    if verbose
        println("\\n" * "="^70)
        println("EstimationToolkit.jl Benchmarks")
        println("="^70)
        println("Steps per filter: $n_steps")
        println("State dimension: 4")
        println("Measurement dimension: 2")
        println("="^70 * "\\n")
    end
    
    results = Dict{String, Float64}()
    
    println(@sprintf("%-35s | %12s | %12s", "Filter", "Time (ms)", "Relative"))
    println("-"^70)
    
    baseline_time = 0.0
    
    for (i, (name, filter)) in enumerate(filters)
        time_per_step = benchmark_filter(filter, dynamics, measurement, n_steps)
        results[name] = time_per_step
        
        if i == 1
            baseline_time = time_per_step
            println(@sprintf("%-35s | %12.4f | %12s", name, time_per_step, "baseline"))
        else
            relative = time_per_step / baseline_time
            println(@sprintf("%-35s | %12.4f | %12.2fx", name, time_per_step, relative))
        end
    end
    
    println("="^70)
    
    if verbose
        println("\\nNotes:")
        println("  - Times are per predict-update cycle")
        println("  - StaticArrays used for small states")
        println("  - Particle filters scale with particle count")
        println("  - Ensemble filters would show better scaling at higher dimensions")
    end
    
    return results
end

"""
    benchmark_state_dimensions()

Test how filter performance scales with state dimension.
"""
function benchmark_state_dimensions()
    println("\\n" * "="^70)
    println("State Dimension Scaling Benchmark")
    println("="^70 * "\\n")
    
    dimensions = [2, 4, 8, 16]
    n_steps = 500
    
    println(@sprintf("%-25s |", "Filter"))
    for dim in dimensions
        @printf(" n=%2d |", dim)
    end
    println()
    println("-"^70)
    
    for filter_type in ["KalmanFilter", "ExtendedKalmanFilter", "UnscentedKalmanFilter"]
        @printf("%-25s |", filter_type)
        
        for dim in dimensions
            # Create appropriately sized initial conditions
            x0 = zeros(dim)
            P0 = Matrix(1.0I, dim, dim)
            
            # Simple linear dynamics
            dynamics = let
                struct ScalableDynamics{N} <: AbstractDynamicsModel{N, 0, N}
                    q::Float64
                end
                EstimationToolkit.propagate(m::ScalableDynamics{N}, x, u, dt) where {N} =
                    (Matrix(1.0I, N, N) + 0.1*dt*Matrix(1.0I, N, N)) * x
                EstimationToolkit.process_noise_cov(m::ScalableDynamics{N}, dt) where {N} =
                    m.q * dt * Matrix(1.0I, N, N)
                ScalableDynamics{dim}(0.1)
            end
            
            # Measure first two components
            measurement = let
                struct ScalableMeasurement{N} <: AbstractMeasurementModel{N, 2, 2}
                    r::Float64
                end
                EstimationToolkit.measure(m::ScalableMeasurement{N}, x) where {N} = x[1:2]
                EstimationToolkit.measurement_noise_cov(m::ScalableMeasurement{N}) where {N} =
                    m.r^2 * Matrix(1.0I, 2, 2)
                ScalableMeasurement{dim}(0.5)
            end
            
            # Create filter
            filter = if filter_type == "KalmanFilter"
                KalmanFilter(x0, P0)
            elseif filter_type == "ExtendedKalmanFilter"
                ExtendedKalmanFilter(x0, P0)
            else  # UKF
                UnscentedKalmanFilter(x0, P0)
            end
            
            # Benchmark
            time_per_step = benchmark_filter(filter, dynamics, measurement, n_steps)
            @printf(" %5.2f |", time_per_step)
        end
        println()
    end
    
    println("="^70)
    println("Note: Times in ms per predict-update cycle")
end

# Run benchmarks if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running EstimationToolkit.jl benchmarks...\\n")
    
    # Main benchmark suite
    run_benchmarks(n_steps=1000)
    
    # Dimension scaling
    # benchmark_state_dimensions()  # Uncomment for dimension scaling study
    
    println("\\n\\nBenchmarks complete!")
end
