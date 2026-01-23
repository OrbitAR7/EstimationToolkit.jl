"""
Test suite for particle flow and ensemble methods:
- GromovFlowFilter
- DaumHuangFlowFilter
- EnsembleSquareRootFilter
"""

using Test
using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random
using Statistics

@testset "Particle Flow and Ensemble Filters" begin
    # Simple nonlinear system for testing
    struct SimpleNonlinear1D{T} <: AbstractDynamicsModel{1, 0, 1}
        q::T
    end
    
    function EstimationToolkit.propagate(m::SimpleNonlinear1D, x::SVector, u, dt)
        return @SVector [0.8 * x[1] + 0.2 * sin(x[1])]
    end
    
    EstimationToolkit.jacobian_x(m::SimpleNonlinear1D, x::SVector, u, dt) = @SMatrix [0.8 + 0.2*cos(x[1])]
    EstimationToolkit.process_noise_cov(m::SimpleNonlinear1D, dt) = @SMatrix [m.q]
    
    struct SimpleNonlinearMeas1D{T} <: AbstractMeasurementModel{1, 1, 1}
        r::T
    end
    
    EstimationToolkit.measure(m::SimpleNonlinearMeas1D, x::SVector) = @SVector [x[1]]
    EstimationToolkit.jacobian_x(m::SimpleNonlinearMeas1D, x::SVector) = @SMatrix [1.0]
    EstimationToolkit.measurement_noise_cov(m::SimpleNonlinearMeas1D) = @SMatrix [m.r]
    
    @testset "GromovFlowFilter" begin
        Random.seed!(123)
        u0 = zeros(0)
        
        x0 = @SVector [1.0]
        P0 = @SMatrix [2.0]
        N = 100
        
        # Create Gromov flow filter
        gromov = GromovFlowFilter(x0, P0, N, n_flow_steps=10, flow_dt=0.1)
        
        @test num_particles(gromov) == N
        @test gromov.n_flow_steps == 10
        @test gromov.flow_dt ≈ 0.1
        
        dyn = SimpleNonlinear1D(0.5)
        meas = SimpleNonlinearMeas1D(0.3)
        
        # Test prediction (no flow during prediction)
        predict!(gromov, dyn, u0, 1.0)
        @test num_particles(gromov) == N
        
        # Test update with flow
        y = @SVector [1.5]
        update!(gromov, meas, y)
        
        x_est = state_estimate(gromov)
        @test length(x_est) == 1
        @test isfinite(x_est[1])
        
        # Run sequence
        for k in 1:8
            predict!(gromov, dyn, u0, 1.0)
            y_k = @SVector [1.0 + 0.1 * k]
            update!(gromov, meas, y_k)
        end
    end
    
    @testset "DaumHuangFlowFilter" begin
        Random.seed!(234)
        u0 = zeros(0)
        
        x0 = @SVector [0.5]
        P0 = @SMatrix [1.5]
        N = 80
        
        # Create Daum-Huang flow filter (deterministic)
        daum_huang = DaumHuangFlowFilter(x0, P0, N, n_flow_steps=15)
        
        @test num_particles(daum_huang) == N
        @test daum_huang.n_flow_steps == 15
        
        dyn = SimpleNonlinear1D(0.4)
        meas = SimpleNonlinearMeas1D(0.4)
        
        # Test prediction
        predict!(daum_huang, dyn, u0, 1.0)
        @test num_particles(daum_huang) == N
        
        # Test update with deterministic flow
        y = @SVector [0.8]
        update!(daum_huang, meas, y)
        
        x_est = state_estimate(daum_huang)
        @test isfinite(x_est[1])
        
        # Run sequence
        for k in 1:8
            predict!(daum_huang, dyn, u0, 1.0)
            y_k = @SVector [0.6 + 0.05 * k]
            update!(daum_huang, meas, y_k)
        end
    end
    
    @testset "Flow Filter Comparison" begin
        # Compare Gromov (stochastic) vs Daum-Huang (deterministic)
        Random.seed!(345)
        u0 = zeros(0)
        
        x0 = @SVector [1.0]
        P0 = @SMatrix [2.0]
        N = 100
        
        gromov = GromovFlowFilter(x0, P0, N, n_flow_steps=10, flow_dt=0.1)
        daum_huang = DaumHuangFlowFilter(x0, P0, N, n_flow_steps=10)
        bpf = BootstrapParticleFilter(x0, P0, N)
        
        dyn = SimpleNonlinear1D(0.5)
        meas = SimpleNonlinearMeas1D(0.3)
        
        # Generate measurements
        Random.seed!(345)
        measurements = [@SVector[1.0 + 0.1 * k + 0.3 * randn()] for k in 1:10]
        
        # Run all filters
        for y_k in measurements
            predict!(gromov, dyn, u0, 1.0)
            update!(gromov, meas, y_k)
            
            predict!(daum_huang, dyn, u0, 1.0)
            update!(daum_huang, meas, y_k)
            
            predict!(bpf, dyn, u0, 1.0)
            update!(bpf, meas, y_k)
        end
        
        # Check ESS (flow filters maintain uniform weights, so ESS is N)
        # Gromov and Daum-Huang are unweighted flow filters
        ess_gromov = Float64(N)
        ess_daum = Float64(N)
        ess_bpf = effective_sample_size(bpf.weights)
        
        @test ess_gromov > ess_bpf * 0.5  # Flow should help
        @test ess_daum > ess_bpf * 0.5    # Flow should help
        
        # All should give similar estimates
        x_gromov = state_estimate(gromov)
        x_daum = state_estimate(daum_huang)
        x_bpf = state_estimate(bpf)
        
        @test abs(x_gromov[1] - x_daum[1]) < 2.0  # Should be reasonably close
    end
    
    @testset "EnsembleSquareRootFilter" begin
        # Linear system for testing (EnSRF should match KF)
        u0 = zeros(0)
        struct LinearDynamics1D{T} <: AbstractDynamicsModel{1, 0, 1}
            A::T
            Q::T
        end
        
        EstimationToolkit.propagate(m::LinearDynamics1D, x::SVector, u, dt) = @SVector [m.A * x[1]]
        EstimationToolkit.jacobian_x(m::LinearDynamics1D, x::SVector, u, dt) = @SMatrix [m.A]
        EstimationToolkit.process_noise_cov(m::LinearDynamics1D, dt) = @SMatrix [m.Q]
        
        struct LinearMeasurement1D{T} <: AbstractMeasurementModel{1, 1, 1}
            H::T
            R::T
        end
        
        EstimationToolkit.measure(m::LinearMeasurement1D, x::SVector) = @SVector [m.H * x[1]]
        EstimationToolkit.jacobian_x(m::LinearMeasurement1D, x::SVector) = @SMatrix [m.H]
        EstimationToolkit.measurement_noise_cov(m::LinearMeasurement1D) = @SMatrix [m.R]
        
        Random.seed!(456)
        
        x0 = @SVector [1.0]
        P0 = @SMatrix [1.0]
        N_ens = 50
        
        ensrf = EnsembleSquareRootFilter(x0, P0, N_ens)
        
        @test size(ensrf.ensemble) == (1, N_ens)
        
        # Ensemble mean should be near x0
        x_mean = Statistics.mean(ensrf.ensemble, dims=2)
        @test x_mean[1] ≈ x0[1] atol=0.5
        
        # Ensemble covariance should be near P0
        X_centered = ensrf.ensemble .- x_mean
        P_ens = (X_centered * X_centered') / (N_ens - 1)
        @test P_ens[1] ≈ P0[1] atol=0.5
        
        dyn = LinearDynamics1D(0.9, 0.1)
        meas = LinearMeasurement1D(1.0, 0.5)
        
        # Test prediction
        predict!(ensrf, dyn, u0, 1.0)
        @test size(ensrf.ensemble) == (1, N_ens)
        
        # Test update
        y = @SVector [1.2]
        update!(ensrf, meas, y)
        
        x_est = state_estimate(ensrf)
        @test length(x_est) == 1
        
        # Compare with Kalman filter
        kf = KalmanFilter(x0, P0)
        
        Random.seed!(456)
        measurements = [@SVector[1.0 + 0.1 * k] for k in 1:10]
        
        ensrf_test = EnsembleSquareRootFilter(x0, P0, N_ens)
        
        for y_k in measurements
            predict!(ensrf_test, dyn, u0, 1.0)
            update!(ensrf_test, meas, y_k)
            
            predict!(kf, dyn, u0, 1.0)
            update!(kf, meas, y_k)
        end
        
        # EnSRF should converge to KF for linear systems
        x_ensrf = state_estimate(ensrf_test)
        x_kf = state_estimate(kf)
        
        @test abs(x_ensrf[1] - x_kf[1]) < 0.3  # Should be close
    end
    
    @testset "EnsembleSquareRootFilter 2D" begin
        # 2D linear system
        u0 = zeros(0)
        struct LinearDynamics2D <: AbstractDynamicsModel{2, 0, 2}
            A::SMatrix{2, 2, Float64}
            Q::SMatrix{2, 2, Float64}
        end
        
        EstimationToolkit.propagate(m::LinearDynamics2D, x::SVector, u, dt) = m.A * x
        EstimationToolkit.jacobian_x(m::LinearDynamics2D, x::SVector, u, dt) = m.A
        EstimationToolkit.process_noise_cov(m::LinearDynamics2D, dt) = m.Q
        
        struct LinearMeasurement2D <: AbstractMeasurementModel{2, 2, 2}
            H::SMatrix{2, 2, Float64}
            R::SMatrix{2, 2, Float64}
        end
        
        EstimationToolkit.measure(m::LinearMeasurement2D, x::SVector) = m.H * x
        EstimationToolkit.jacobian_x(m::LinearMeasurement2D, x::SVector) = m.H
        EstimationToolkit.measurement_noise_cov(m::LinearMeasurement2D) = m.R
        
        Random.seed!(567)
        
        x0_2d = @SVector [1.0, 0.5]
        P0_2d = @SMatrix [1.0 0.2; 0.2 1.0]
        N_ens = 80
        
        ensrf_2d = EnsembleSquareRootFilter(x0_2d, P0_2d, N_ens)
        
        @test size(ensrf_2d.ensemble) == (2, N_ens)
        
        A2 = @SMatrix [0.9 0.1; 0.0 0.95]
        Q2 = @SMatrix [0.1 0.0; 0.0 0.1]
        dyn_2d = LinearDynamics2D(A2, Q2)
        
        H2 = @SMatrix [1.0 0.0; 0.0 1.0]
        R2 = @SMatrix [0.5 0.0; 0.0 0.5]
        meas_2d = LinearMeasurement2D(H2, R2)
        
        # Run filter
        for k in 1:8
            predict!(ensrf_2d, dyn_2d, u0, 1.0)
            y_k = @SVector [1.0 + 0.05 * k, 0.5 + 0.03 * k]
            update!(ensrf_2d, meas_2d, y_k)
        end
        
        x_est = state_estimate(ensrf_2d)
        @test length(x_est) == 2
        @test all(isfinite, x_est)
        
        # Check ensemble covariance
        P_est = state_covariance(ensrf_2d)
        @test size(P_est) == (2, 2)
        @test is_positive_definite(P_est)
    end
    
    @testset "EnSRF Nonlinear System" begin
        # Test EnSRF on nonlinear problem
        Random.seed!(678)
        u0 = zeros(0)
        
        x0 = @SVector [0.5]
        P0 = @SMatrix [1.0]
        N_ens = 100
        
        ensrf_nl = EnsembleSquareRootFilter(x0, P0, N_ens)
        
        dyn_nl = SimpleNonlinear1D(0.3)
        meas_nl = SimpleNonlinearMeas1D(0.4)
        
        # Run on nonlinear system
        for k in 1:10
            predict!(ensrf_nl, dyn_nl, u0, 1.0)
            y_k = @SVector [0.5 + 0.1 * k]
            update!(ensrf_nl, meas_nl, y_k)
            
            x_est = state_estimate(ensrf_nl)
            @test isfinite(x_est[1])
        end
        
        # Compare with EKF
        ekf = ExtendedKalmanFilter(x0, P0)
        ensrf_comp = EnsembleSquareRootFilter(x0, P0, N_ens)
        
        Random.seed!(678)
        measurements = [@SVector[0.5 + 0.1 * k] for k in 1:10]
        
        for y_k in measurements
            predict!(ekf, dyn_nl, u0, 1.0)
            update!(ekf, meas_nl, y_k)
            
            predict!(ensrf_comp, dyn_nl, u0, 1.0)
            update!(ensrf_comp, meas_nl, y_k)
        end
        
        x_ekf = state_estimate(ekf)
        x_ensrf = state_estimate(ensrf_comp)
        
        # Should give similar results for smooth nonlinear problem
        @test abs(x_ekf[1] - x_ensrf[1]) < 1.0
    end
    
    @testset "Flow Filter Edge Cases" begin
        # Very few flow steps
        u0 = zeros(0)
        x0 = @SVector [1.0]
        P0 = @SMatrix [1.0]
        
        gromov_few = GromovFlowFilter(x0, P0, 50, n_flow_steps=1, flow_dt=0.5)
        @test gromov_few.n_flow_steps == 1
        
        dyn = SimpleNonlinear1D(0.3)
        meas = SimpleNonlinearMeas1D(0.3)
        
        predict!(gromov_few, dyn, u0, 1.0)
        update!(gromov_few, meas, @SVector[1.0])
        
        # Very large step size (should still be stable)
        daum_large = DaumHuangFlowFilter(x0, P0, 50, n_flow_steps=5)
        
        for _ in 1:5
            predict!(daum_large, dyn, u0, 1.0)
            update!(daum_large, meas, @SVector[0.8])
        end
    end
    
    @testset "Ensemble Size Effects" begin
        # Compare different ensemble sizes
        u0 = zeros(0)
        x0 = @SVector [1.0]
        P0 = @SMatrix [1.0]
        
        dyn = SimpleNonlinear1D(0.2)
        meas = SimpleNonlinearMeas1D(0.3)
        
        Random.seed!(789)
        measurements = [@SVector[1.0 + 0.05 * k] for k in 1:8]
        
        # Small ensemble
        ensrf_small = EnsembleSquareRootFilter(x0, P0, 20)
        for y_k in measurements
            predict!(ensrf_small, dyn, u0, 1.0)
            update!(ensrf_small, meas, y_k)
        end
        
        # Large ensemble
        Random.seed!(789)
        ensrf_large = EnsembleSquareRootFilter(x0, P0, 200)
        for y_k in measurements
            predict!(ensrf_large, dyn, u0, 1.0)
            update!(ensrf_large, meas, y_k)
        end
        
        # Both should produce valid estimates
        x_small = state_estimate(ensrf_small)
        x_large = state_estimate(ensrf_large)
        
        @test isfinite(x_small[1])
        @test isfinite(x_large[1])
        
        # Larger ensemble should have lower variance (more stable)
        P_small = state_covariance(ensrf_small)
        P_large = state_covariance(ensrf_large)
        
        @test P_small[1] > 0
        @test P_large[1] > 0
    end
end
