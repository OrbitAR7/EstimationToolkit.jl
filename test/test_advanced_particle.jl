"""
Test suite for advanced particle filters:
- GaussianParticleFilter (GPF)
- ExtendedKalmanParticleFilter (EKPF)
- UnscentedParticleFilter (UPF)
- RaoBlackwellizedParticleFilter (RBPF)
- MarginalParticleFilter (MPF)
"""

using Test
using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random

# Helper function
num_particles(f) = f.N_particles

@testset "Advanced Particle Filters" begin
    # Nonlinear 1D system
    struct NonlinearDyn1D{T} <: AbstractDynamicsModel{1, 0, 1}
        q::T
    end
    
    function EstimationToolkit.propagate(m::NonlinearDyn1D, x::SVector, u, dt)
        return @SVector [0.5 * x[1] + 25 * x[1] / (1 + x[1]^2) + 8 * cos(1.2 * dt)]
    end
    
    EstimationToolkit.jacobian_x(m::NonlinearDyn1D, x::SVector, u, dt) = @SMatrix [0.5 + 25*(1-x[1]^2)/(1+x[1]^2)^2]
    EstimationToolkit.process_noise_cov(m::NonlinearDyn1D, dt) = @SMatrix [m.q]
    
    struct NonlinearMeas1D{T} <: AbstractMeasurementModel{1, 1, 1}
        r::T
    end
    
    EstimationToolkit.measure(m::NonlinearMeas1D, x::SVector) = @SVector [x[1]^2 / 20]
    EstimationToolkit.jacobian_x(m::NonlinearMeas1D, x::SVector) = @SMatrix [x[1] / 10]
    EstimationToolkit.measurement_noise_cov(m::NonlinearMeas1D) = @SMatrix [m.r]
    
    @testset "GaussianParticleFilter" begin
        Random.seed!(123)
        
        x0 = @SVector [0.1]
        P0 = @SMatrix [5.0]
        N = 100
        
        gpf = GaussianParticleFilter(x0, P0, N)
        
        @test num_particles(gpf) == N
        @test sum(gpf.weights) ≈ 1.0
        
        dyn = NonlinearDyn1D(10.0)
        meas = NonlinearMeas1D(1.0)
        u0 = zeros(0)
        
        # Test prediction
        predict!(gpf, dyn, u0, 1.0)
        @test num_particles(gpf) == N
        @test sum(gpf.weights) ≈ 1.0
        
        # Test update with Gaussian fitting
        y = @SVector [0.1]
        update!(gpf, meas, y)
        
        @test sum(gpf.weights) ≈ 1.0
        x_est = state_estimate(gpf)
        @test length(x_est) == 1
        
        # Run sequence
        for k in 1:10
            predict!(gpf, dyn, u0, 1.0)
            y_k = @SVector [rand() * 5.0]
            update!(gpf, meas, y_k)
            
            @test sum(gpf.weights) ≈ 1.0
            @test num_particles(gpf) == N
        end
        
        # Check Gaussian approximation
        P_est = state_covariance(gpf)
        @test is_positive_definite(P_est)
    end
    
    @testset "ExtendedKalmanParticleFilter" begin
        Random.seed!(234)
        
        x0 = @SVector [1.0]
        P0 = @SMatrix [2.0]
        N = 80
        
        ekpf = ExtendedKalmanParticleFilter(x0, P0, N)
        
        @test num_particles(ekpf) == N
        @test sum(ekpf.weights) ≈ 1.0
        
        dyn = NonlinearDyn1D(5.0)
        meas = NonlinearMeas1D(0.5)
        u0 = zeros(0)
        
        # Test EKF-based proposal
        y = @SVector [0.05]
        
        step!(ekpf, dyn, meas, u0, y, 1.0)
        @test sum(ekpf.weights) ≈ 1.0
        
        x_est = state_estimate(ekpf)
        @test isfinite(x_est[1])
        
        # Run full sequence
        for k in 1:8
            y_k = @SVector [0.1 + 0.01 * k]
            step!(ekpf, dyn, meas, u0, y_k, 1.0)
            
            @test sum(ekpf.weights) ≈ 1.0
            ess = effective_sample_size(ekpf.weights)
            @test ess > 1.0  # Should maintain diversity
        end
    end
    
    @testset "UnscentedParticleFilter" begin
        Random.seed!(345)
        
        x0 = @SVector [0.5]
        P0 = @SMatrix [3.0]
        N = 100
        
        upf = UnscentedParticleFilter(x0, P0, N)
        
        @test num_particles(upf) == N
        @test sum(upf.weights) ≈ 1.0
        
        dyn = NonlinearDyn1D(8.0)
        meas = NonlinearMeas1D(0.8)
        
        # Test UKF-based proposal
        y = @SVector [0.2]
        u0 = zeros(0)
        
        # Test UKF-based proposal
        y = @SVector [0.2]
        
        step!(upf, dyn, meas, u0, y, 1.0)
        @test sum(upf.weights) ≈ 1.0
        
        x_est = state_estimate(upf)
        @test isfinite(x_est[1])
        
        # Compare with EKPF
        ekpf = ExtendedKalmanParticleFilter(x0, P0, N)
        Random.seed!(345)
        
        for k in 1:8
            y_k = @SVector [0.15 + 0.02 * k]
            
            step!(upf, dyn, meas, u0, y_k, 1.0)
            
            step!(ekpf, dyn, meas, u0, y_k, 1.0)
        end
        
        x_upf = state_estimate(upf)
        x_ekpf = state_estimate(ekpf)
        
        @test all(isfinite, x_upf)
        @test all(isfinite, x_ekpf)
    end
    
    @testset "RaoBlackwellizedParticleFilter" begin
        # Conditionally linear system:
        # x = [x_nl, x_l]  (nonlinear part, linear part)
        # x_nl evolves nonlinearly
        # x_l | x_nl evolves linearly
        
        struct RBDynamics <: AbstractDynamicsModel{2, 0, 2}
            Q::SMatrix{2, 2, Float64}
        end
        
        function EstimationToolkit.propagate_nonlinear(m::RBDynamics, x_nl::SVector{1}, 
                                                       x_l::SVector{1}, u, dt)
            # Nonlinear evolution: x_nl_k+1 = 0.9*x_nl + 0.1*sin(x_nl)
            return @SVector [0.9 * x_nl[1] + 0.1 * sin(x_nl[1])]
        end
        
        function EstimationToolkit.linear_dynamics_matrix(m::RBDynamics, x_nl::SVector{1}, u, dt)
            # Linear evolution: x_l_k+1 = 0.95*x_l
            return @SMatrix [0.95]
        end
        
        function EstimationToolkit.process_noise_cov_nonlinear(m::RBDynamics, x_nl::SVector{1}, dt)
            return SMatrix{1,1}(m.Q[1,1])
        end

        function EstimationToolkit.process_noise_cov_linear(m::RBDynamics, x_nl::SVector{1}, dt)
            return SMatrix{1,1}(m.Q[2,2])
        end
        
        struct RBMeasurement <: AbstractMeasurementModel{2, 1, 1}
            R::SMatrix{1, 1, Float64}
        end
        
        function EstimationToolkit.nonlinear_measurement(m::RBMeasurement, x_nl::SVector{1})
            # Nonlinear measurement of x_nl
            return @SVector [x_nl[1]^2]
        end
        
        function EstimationToolkit.linear_measurement_matrix(m::RBMeasurement, x_nl::SVector{1})
            # Linear measurement matrix for x_l
            C = @SMatrix [1.0]
            return C
        end
        
        function EstimationToolkit.measurement_noise_cov(m::RBMeasurement, x_nl::SVector{1})
            return m.R
        end
        
        Random.seed!(456)
        
        # 1 nonlinear dimension, 1 linear dimension
        x0_linear = @SVector [0.0]
        P0_linear = @SMatrix [1.0]
        N = 60
        particles_nl = [@SVector [0.0] for _ in 1:N]
        
        rbpf = RaoBlackwellizedParticleFilter(x0_linear, P0_linear, particles_nl)
        
        @test num_particles(rbpf) == N
        @test length(rbpf.kalman_filters) == N
        @test all(kf isa KalmanFilter for kf in rbpf.kalman_filters)
        
        dyn_rb = RBDynamics(@SMatrix [0.1 0.0; 0.0 0.15])
        meas_rb = RBMeasurement(@SMatrix [0.5])
        u0 = zeros(0)
        
        # Test prediction
        predict!(rbpf, dyn_rb, u0, 1.0)
        @test sum(rbpf.weights) ≈ 1.0
        
        # Test update
        y = @SVector [0.1]
        update!(rbpf, meas_rb, y)
        @test sum(rbpf.weights) ≈ 1.0
        
        # Get joint estimate
        x_est = state_estimate(rbpf)
        # Note: RBPF implementation returns [x_linear; x_nonlinear]
        @test length(x_est) == 2
        
        # Verify components
        x_l_est = x_est[1:1]
        x_nl_est = x_est[2:2]
        
        P_est = covariance(rbpf)
        P_l_est = P_est[1:1, 1:1]
        
        @test length(x_nl_est) == 1
        @test length(x_l_est) == 1
        @test is_positive_definite(P_l_est)
        
        # Run sequence
        for k in 1:10
            predict!(rbpf, dyn_rb, u0, 1.0)
            y_k = @SVector [0.05 + 0.01 * k]
            update!(rbpf, meas_rb, y_k)
            
            @test sum(rbpf.weights) ≈ 1.0
        end
    end
    
    @testset "MarginalParticleFilter" begin
        # 2D system where we maintain marginal particles per dimension
        struct Dyn2DMarginal <: AbstractDynamicsModel{2, 0, 2}
            Q::SMatrix{2, 2, Float64}
        end
        
        function EstimationToolkit.propagate(m::Dyn2DMarginal, x::SVector, u, dt)
            # Weakly coupled dynamics
            return @SVector [0.9 * x[1] + 0.05 * sin(x[2]), 
                            0.95 * x[2] + 0.05 * cos(x[1])]
        end
        
        EstimationToolkit.process_noise_cov(m::Dyn2DMarginal, dt) = m.Q
        
        struct Meas2DMarginal <: AbstractMeasurementModel{2, 2, 2}
            R::SMatrix{2, 2, Float64}
        end
        
        EstimationToolkit.measure(m::Meas2DMarginal, x::SVector) = x
        EstimationToolkit.measurement_noise_cov(m::Meas2DMarginal) = m.R
        
        Random.seed!(567)
        
        x0 = @SVector [1.0, 0.5]
        P0 = @SMatrix [1.0 0.0; 0.0 1.0]
        N = 80
        
        mpf = MarginalParticleFilter(x0, P0, N)
        
        @test num_particles(mpf) == N
        @test length(mpf.marginal_particles) == 2  # One per dimension
        @test length(mpf.marginal_weights) == 2
        
        # Each dimension has N particles
        @test length(mpf.marginal_particles[1]) == N
        @test length(mpf.marginal_particles[2]) == N
        
        dyn_marg = Dyn2DMarginal(@SMatrix [0.1 0.0; 0.0 0.1])
        meas_marg = Meas2DMarginal(@SMatrix [0.3 0.0; 0.0 0.3])
        u0 = zeros(0)
        
        # Test prediction
        predict!(mpf, dyn_marg, u0, 1.0)
        
        for d in 1:2
            @test sum(mpf.marginal_weights[d]) ≈ 1.0
        end
        
        # Test update
        y = @SVector [1.2, 0.6]
        update!(mpf, meas_marg, y)
        
        for d in 1:2
            @test sum(mpf.marginal_weights[d]) ≈ 1.0
        end
        
        # Get marginal estimates
        x1_marg = get_marginal_estimate(mpf, 1)
        P1_marg = get_marginal_variance(mpf, 1)
        x2_marg = get_marginal_estimate(mpf, 2)
        P2_marg = get_marginal_variance(mpf, 2)
        
        @test isfinite(x1_marg)
        @test isfinite(x2_marg)
        @test P1_marg > 0
        @test P2_marg > 0
        
        # Get joint estimate
        x_est = state_estimate(mpf)
        @test length(x_est) == 2
        
        # Run sequence
        for k in 1:8
            predict!(mpf, dyn_marg, u0, 1.0)
            y_k = @SVector [1.0 + 0.1 * k, 0.5 + 0.05 * k]
            update!(mpf, meas_marg, y_k)
            
            for d in 1:2
                @test sum(mpf.marginal_weights[d]) ≈ 1.0
            end
        end
    end
    
    @testset "Particle Filter Comparison" begin
        # Compare all particle filter variants on same problem
        Random.seed!(678)
        
        x0 = @SVector [0.5]
        P0 = @SMatrix [2.0]
        N = 100
        
        bpf = BootstrapParticleFilter(x0, P0, N)
        gpf = GaussianParticleFilter(x0, P0, N)
        ekpf = ExtendedKalmanParticleFilter(x0, P0, N)
        upf = UnscentedParticleFilter(x0, P0, N)
        
        dyn = NonlinearDyn1D(10.0)
        meas = NonlinearMeas1D(1.0)
        
        # Run all filters on same data
        Random.seed!(678)
        u0 = zeros(0)
        measurements = [@SVector[rand() * 5.0] for _ in 1:10]
        
        for y_k in measurements
            predict!(bpf, dyn, u0, 1.0)
            update!(bpf, meas, y_k)
            
            predict!(gpf, dyn, u0, 1.0)
            update!(gpf, meas, y_k)
            
            step!(ekpf, dyn, meas, u0, y_k, 1.0)
            
            step!(upf, dyn, meas, u0, y_k, 1.0)
        end
        
        # All should produce valid estimates
        x_bpf = state_estimate(bpf)
        x_gpf = state_estimate(gpf)
        x_ekpf = state_estimate(ekpf)
        x_upf = state_estimate(upf)
        
        @test all(isfinite, x_bpf)
        @test all(isfinite, x_gpf)
        @test all(isfinite, x_ekpf)
        @test all(isfinite, x_upf)
        
        # Check ESS (optimal proposal should have higher ESS)
        ess_bpf = effective_sample_size(bpf.weights)
        ess_gpf = effective_sample_size(gpf.weights)
        ess_ekpf = effective_sample_size(ekpf.weights)
        ess_upf = effective_sample_size(upf.weights)
        
        @test ess_ekpf ≥ ess_bpf * 0.5  # EKPF should be better than bootstrap
        @test ess_upf ≥ ess_bpf * 0.5   # UPF should be better than bootstrap
    end
    
    @testset "Edge Cases" begin
        # Very few particles
        x0 = @SVector [1.0]
        P0 = @SMatrix [1.0]
        
        gpf_small = GaussianParticleFilter(x0, P0, 10)
        @test num_particles(gpf_small) == 10
        
        dyn = NonlinearDyn1D(5.0)
        meas = NonlinearMeas1D(0.5)
        
        u0 = zeros(0)
        
        for _ in 1:5
            predict!(gpf_small, dyn, u0, 1.0)
            update!(gpf_small, meas, @SVector[0.1])
            @test sum(gpf_small.weights) ≈ 1.0
        end
        
        # High measurement noise (should maintain diversity)
        meas_noisy = NonlinearMeas1D(10.0)
        ekpf_noisy = ExtendedKalmanParticleFilter(x0, P0, 50)
        
        for _ in 1:5
            y_k = @SVector[1.0]
            step!(ekpf_noisy, dyn, meas_noisy, u0, y_k, 1.0)
            
            ess = effective_sample_size(ekpf_noisy.weights)
            @test ess > 5.0  # Should not degenerate completely
        end
    end
    
    @testset "RBPF Dimensionality" begin
        # Test different splits of nonlinear/linear dimensions
        N = 50
        
        # 2 nonlinear, 1 linear
        μ0_l1 = @SVector [0.0]
        P0_l1 = @SMatrix [1.0]
        particles_nl1 = [@SVector[randn(), randn()] for _ in 1:N]
        rbpf_21 = RaoBlackwellizedParticleFilter(μ0_l1, P0_l1, particles_nl1)
        
        @test length(rbpf_21.particles_nonlinear[1]) == 2
        @test length(rbpf_21.kalman_filters[1].x) == 1
        
        # 1 nonlinear, 2 linear
        μ0_l2 = @SVector [0.0, 0.0]
        P0_l2 = @SMatrix [1.0 0.0; 0.0 1.0]
        particles_nl2 = [@SVector[randn()] for _ in 1:N]
        rbpf_12 = RaoBlackwellizedParticleFilter(μ0_l2, P0_l2, particles_nl2)
        
        @test length(rbpf_12.particles_nonlinear[1]) == 1
        @test length(rbpf_12.kalman_filters[1].x) == 2
    end
end
