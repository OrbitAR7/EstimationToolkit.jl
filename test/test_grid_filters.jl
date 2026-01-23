"""
Test suite for grid-based filters:
- PointMassFilter (PMF)
- AdaptiveGridFilter
- MarginalizedPointMassFilter
"""

using Test
using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random
using Distributions

@testset "Grid-Based Filters" begin
    # Simple 1D nonlinear system
    struct Nonlinear1DDynamics{T} <: AbstractDynamicsModel{1, 0, 1}
        q::T
    end
    
    function EstimationToolkit.propagate(m::Nonlinear1DDynamics, x::SVector, u, dt)
        # x_k+1 = 0.9*x_k + 0.1*x_k^2 + noise
        return @SVector [0.9 * x[1] + 0.1 * x[1]^2]
    end
    
    EstimationToolkit.process_noise_cov(m::Nonlinear1DDynamics, dt) = @SMatrix [m.q]
    
    struct Nonlinear1DMeasurement{T} <: AbstractMeasurementModel{1, 1, 1}
        r::T
    end
    
    function EstimationToolkit.measure(m::Nonlinear1DMeasurement, x::SVector)
        # y = x^2 + noise (strongly nonlinear)
        return @SVector [x[1]^2]
    end
    
    EstimationToolkit.measurement_noise_cov(m::Nonlinear1DMeasurement) = @SMatrix [m.r]
    
    @testset "PointMassFilter" begin
        # Create grid
        bounds = [(Float64(-3.0), Float64(3.0))]
        n_points_per_dim = [50]
        x0 = @SVector [0.0]
        
        pmf = PointMassFilter(bounds, n_points_per_dim, x0)
        
        @test length(pmf.grid_points) == 50
        @test all(-3.0 ≤ g[1] ≤ 3.0 for g in pmf.grid_points)
        @test issorted([g[1] for g in pmf.grid_points])
        
        @test length(pmf.probabilities) == 50
        @test sum(pmf.probabilities) ≈ 1.0
        
        # Check grid statistics (use direct computation)
        mean_x = sum(pmf.grid_points[i][1] * pmf.probabilities[i] for i in 1:length(pmf.grid_points))
        @test isfinite(mean_x)
        
        # Test prediction
        dyn = Nonlinear1DDynamics(0.1)
        predict!(pmf, dyn, nothing, 1.0)
        
        @test sum(pmf.probabilities) ≈ 1.0 atol=1e-6
        @test all(w ≥ 0 for w in pmf.probabilities)
        
        # Test update
        meas = Nonlinear1DMeasurement(0.5)
        y = @SVector [0.25]  # Observation
        
        update!(pmf, meas, y)
        
        @test sum(pmf.probabilities) ≈ 1.0
        x_est = state_estimate(pmf)
        @test length(x_est) == 1
        
        # Check find_closest_grid_point
        test_point = @SVector [0.5]
        idx = find_closest_grid_point(test_point, pmf.grid_points)
        @test 1 ≤ idx ≤ 50
        
        # Full cycle
        for _ in 1:5
            predict!(pmf, dyn, nothing, 1.0)
            y_new = @SVector [rand() * 2.0]
            update!(pmf, meas, y_new)
            @test sum(pmf.probabilities) ≈ 1.0
        end
    end
    
    @testset "PointMassFilter 2D" begin
        # Test with 2D state
        struct Nonlinear2DDynamics <: AbstractDynamicsModel{2, 0, 2}
            Q::SMatrix{2, 2, Float64}
        end
        
        function EstimationToolkit.propagate(m::Nonlinear2DDynamics, x::SVector, u, dt)
            # Simple rotation + nonlinearity
            θ = 0.1
            return @SVector [cos(θ) * x[1] - sin(θ) * x[2], 
                            sin(θ) * x[1] + cos(θ) * x[2] * 0.95]
        end
        
        EstimationToolkit.process_noise_cov(m::Nonlinear2DDynamics, dt) = m.Q
        
        bounds_2d = [(Float64(-2.0), Float64(2.0)), (Float64(-2.0), Float64(2.0))]
        n_points_per_dim_2d = [20, 20]
        x0_2d = @SVector [0.5, 0.5]
        
        pmf_2d = PointMassFilter(bounds_2d, n_points_per_dim_2d, x0_2d)
        
        @test length(pmf_2d.grid_points) == 20^2
        
        dyn_2d = Nonlinear2DDynamics(@SMatrix [0.1 0.0; 0.0 0.1])
        predict!(pmf_2d, dyn_2d, nothing, 1.0)
        
        @test sum(pmf_2d.probabilities) ≈ 1.0 atol=1e-6
    end
    
    @testset "AdaptiveGridFilter" begin
        bounds = [(Float64(-5.0), Float64(5.0))]
        n_points_per_dim_init = [30]
        
        x0 = @SVector [0.0]
        
        agf = AdaptiveGridFilter(bounds, n_points_per_dim_init, x0, 
                                 refine_threshold=0.1, 
                                 coarse_threshold=0.01,
                                 prune_threshold=1e-6,
                                 max_level=3)
        
        @test length(agf.grid_points) == 30
        @test length(agf.probabilities) == 30
        
        dyn = Nonlinear1DDynamics(0.1)
        meas = Nonlinear1DMeasurement(0.3)
        
        # Run filter with adaptation
        initial_points = length(agf.grid_points)
        
        for k in 1:10
            predict!(agf, dyn, nothing, 1.0)
            
            # Create peaked measurement (should trigger refinement)
            y = @SVector [0.1]
            update!(agf, meas, y)
            
            # Check valid probabilities
            @test sum(agf.probabilities) ≈ 1.0 atol=1e-6
        end
        
        # Grid may have adapted
        @test length(agf.grid_points) > 0
    end
    
    # Define Mock Models for Marginalized Estimator
    struct MarginalizedDynamics{T}
        Q::SMatrix{2, 2, T}
    end
    
    struct MarginalizedMeasurement{T}
        R::SMatrix{1, 1, T}
    end
    
    function EstimationToolkit.propagate_nonlinear(model::MarginalizedDynamics, xn, xl, u, dt)
        return xn # Static nonlinear part for simplicity
    end
    
    function EstimationToolkit.linear_dynamics_matrix(model::MarginalizedDynamics, xn, u, dt)
        return @SMatrix [1.0] # Identity linear dynamics
    end
    
    function EstimationToolkit.process_noise_cov_nonlinear(model::MarginalizedDynamics, xn, dt)
        return SMatrix{1,1}(model.Q[1,1])
    end
    
    function EstimationToolkit.process_noise_cov_linear(model::MarginalizedDynamics, xn, dt)
        return SMatrix{1,1}(model.Q[2,2])
    end
    
    function EstimationToolkit.linear_measurement_matrix(model::MarginalizedMeasurement, xn)
        return @SMatrix [1.0]
    end
    
    function EstimationToolkit.nonlinear_measurement(model::MarginalizedMeasurement, xn)
        return @SVector [0.0]
    end
    
    function EstimationToolkit.measurement_noise_cov(model::MarginalizedMeasurement, xn)
        return model.R
    end

    @testset "MarginalizedPointMassFilter" begin
        # Hybrid system test - requires complex API, skip for now
        # TODO: Implement proper marginalized system model
        
        # Basic constructor test
        bounds_nl = [(Float64(-3.0), Float64(3.0))]
        n_points_per_dim_nl = [40]
        
        x_nl = @SVector [0.0]  # x1_nonlinear
        x_lin = @SVector [0.0] # x2_linear
        P_lin = @SMatrix [1.0] # Initial covariance for linear part
        
        # Create filter (1 nonlinear dimension, 1 linear dimension)
        mpf = MarginalizedPointMassFilter(bounds_nl, n_points_per_dim_nl, x_lin, P_lin, x_nl)
        
        @test num_grid_points(mpf) == 40
        @test length(mpf.kalman_filters) == 40
        @test all(kf isa KalmanFilter for kf in mpf.kalman_filters)
        
        dyn_marg = MarginalizedDynamics(@SMatrix [0.1 0.0; 0.0 0.15])
        meas_marg = MarginalizedMeasurement(@SMatrix [0.3])
        
        # Test prediction
        predict!(mpf, dyn_marg, nothing, 1.0)
        @test sum(mpf.probabilities) ≈ 1.0 atol=1e-6
        
        # Test update
        y = @SVector [0.2]
        update!(mpf, meas_marg, y)
        @test sum(mpf.probabilities) ≈ 1.0 atol=1e-6
        
        # Get joint estimate
        x_est = state_estimate(mpf)
        @test length(x_est) == 2  # [x1, x2]
        
        # Run full sequence
        for _ in 1:5
            predict!(mpf, dyn_marg, nothing, 1.0)
            y_new = @SVector [rand() * 1.5]
            update!(mpf, meas_marg, y_new)
            @test sum(mpf.probabilities) ≈ 1.0 atol=1e-6
        end
    end
    
    @testset "Grid Filter Edge Cases" begin
        # Very coarse grid
        bounds_coarse = [(Float64(-1.0), Float64(1.0))]
        n_coarse = [5]
        
        x0 = @SVector [0.0]
        pmf_coarse = PointMassFilter(bounds_coarse, n_coarse, x0)
        
        @test length(pmf_coarse.grid_points) == 5
        
        # Single point grid (degenerate case) - need at least 2 points
        bounds_single = [(Float64(-0.1), Float64(0.1))]
        n_single = [2]  # Minimum allowed
        pmf_single = PointMassFilter(bounds_single, n_single, x0)
        
        @test length(pmf_single.grid_points) == 2
        @test sum(pmf_single.probabilities) ≈ 1.0
        
        # Test with uniform initialization (no x0)
        pmf_uniform = PointMassFilter(bounds_coarse, n_coarse, nothing)
        @test sum(pmf_uniform.probabilities) ≈ 1.0
        @test all(p ≈ 0.2 for p in pmf_uniform.probabilities)  # Uniform over 5 points
    end
    
    @testset "Grid Utilities" begin
        # Test closest point finding
        test_grid = [SVector{1}(-1.0), SVector{1}(0.0), SVector{1}(1.0)]
        @test find_closest_grid_point(SVector{1}(-0.9), test_grid) == 1
        @test find_closest_grid_point(SVector{1}(0.1), test_grid) == 2
        @test find_closest_grid_point(SVector{1}(0.9), test_grid) == 3
    end
end
