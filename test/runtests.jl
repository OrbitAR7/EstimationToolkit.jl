using Test
using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random

@testset "EstimationToolkit.jl" begin
    
    @testset "Core Types" begin
        # GaussianBelief
        μ = @SVector [1.0, 2.0]
        Σ = @SMatrix [1.0 0.1; 0.1 1.0]
        belief = GaussianBelief(μ, Σ)
        
        @test belief.μ == μ
        @test belief.Σ == Σ
        @test length(belief) == 2
        @test eltype(belief) == Float64
        
        # ParticleBelief
        particles = [@SVector[1.0, 2.0], @SVector[1.5, 2.5], @SVector[0.5, 1.5]]
        weights = [0.5, 0.3, 0.2]
        pb = ParticleBelief(particles, weights)
        
        @test num_particles(pb) == 3
        @test sum(pb.weights) ≈ 1.0
    end
    
    @testset "Matrix Operations" begin
        # Positive definiteness check
        P_good = @SMatrix [2.0 0.5; 0.5 2.0]
        @test is_positive_definite(P_good)
        
        P_bad = @SMatrix [1.0 2.0; 2.0 1.0]
        @test !is_positive_definite(P_bad)
        
        # Symmetrization
        A = @SMatrix [1.0 0.1; 0.2 1.0]
        A_sym = make_symmetric(A)
        @test issymmetric(A_sym)
        @test A_sym[1, 2] == A_sym[2, 1]
    end
    
    @testset "Statistics Utilities" begin
        # Effective sample size
        weights_uniform = fill(0.25, 4)
        @test effective_sample_size(weights_uniform) ≈ 4.0
        
        weights_degenerate = [0.99, 0.01, 0.0, 0.0]
        @test effective_sample_size(weights_degenerate) < 2.0
        
        # Sample mean
        particles = [@SVector[1.0, 0.0], @SVector[0.0, 1.0]]
        weights = [0.5, 0.5]
        μ = sample_mean(particles, weights)
        @test μ ≈ @SVector[0.5, 0.5]
    end
    
    @testset "Simple Linear System" begin
        # Define a simple 1D linear system
        struct SimpleLinearDynamics{T} <: AbstractDynamicsModel{1, 0, 1}
            A::T
            Q::T
        end
        
        EstimationToolkit.propagate(m::SimpleLinearDynamics, x::SVector, u, dt) = @SVector [m.A * x[1]]
        EstimationToolkit.jacobian_x(m::SimpleLinearDynamics, x::SVector, u, dt) = @SMatrix [m.A]
        EstimationToolkit.process_noise_cov(m::SimpleLinearDynamics, dt) = @SMatrix [m.Q]
        
        struct SimpleLinearMeasurement{T} <: AbstractMeasurementModel{1, 1, 1}
            H::T
            R::T
        end
        
        EstimationToolkit.measure(m::SimpleLinearMeasurement, x::SVector) = @SVector [m.H * x[1]]
        EstimationToolkit.jacobian_x(m::SimpleLinearMeasurement, x::SVector) = @SMatrix [m.H]
        EstimationToolkit.measurement_noise_cov(m::SimpleLinearMeasurement) = @SMatrix [m.R]
        
        # Create models
        dyn = SimpleLinearDynamics(0.9, 0.1)
        meas = SimpleLinearMeasurement(1.0, 0.5)
        
        # Initial state
        x0 = @SVector [1.0]
        P0 = @SMatrix [1.0]
        
        @testset "Kalman Filter" begin
            kf = KalmanFilter(x0, P0)
            
            # Predict
            predict!(kf, dyn, nothing, 1.0)
            @test kf.x[1] ≈ 0.9  # A * x0
            # For A < 1, P̄ = A*P*A' + Q = 0.9^2 * 1.0 + 0.1 = 0.81 + 0.1 = 0.91
            @test kf.P[1] ≈ 0.91 atol=1e-10
            
            # Update
            y = @SVector [0.8]
            update!(kf, meas, y)
            @test kf.ν !== nothing
            @test innovation(kf) !== nothing
            @test state_estimate(kf) == kf.x
        end
        
        @testset "Extended Kalman Filter" begin
            ekf = ExtendedKalmanFilter(x0, P0)
            
            predict!(ekf, dyn, nothing, 1.0)
            @test ekf.x[1] ≈ 0.9
            
            y = @SVector [0.8]
            update!(ekf, meas, y)
            @test innovation(ekf) !== nothing
        end
        
        @testset "Unscented Kalman Filter" begin
            ukf = UnscentedKalmanFilter(x0, P0)
            
            predict!(ukf, dyn, nothing, 1.0)
            @test ukf.x[1] ≈ 0.9 atol=1e-10  # Should be exact for linear
            
            y = @SVector [0.8]
            update!(ukf, meas, y)
            @test innovation(ukf) !== nothing
        end
    end
    
    @testset "Resampling" begin
        Random.seed!(123)
        
        weights = [0.1, 0.2, 0.3, 0.4]
        N = 100
        
        # Systematic resampling
        indices = systematic_resample(weights, N)
        @test length(indices) == N
        @test all(1 ≤ i ≤ 4 for i in indices)
        
        # Count how many times each index appears
        counts = [count(==(i), indices) for i in 1:4]
        # Should be roughly proportional to weights
        @test counts[4] > counts[1]  # Weight 0.4 should appear more than 0.1
        
        # Stratified resampling
        indices = stratified_resample(weights, N)
        @test length(indices) == N
        
        # Multinomial resampling
        indices = multinomial_resample(weights, N)
        @test length(indices) == N
    end
    
    @testset "Particle Filter" begin
        # Simple 1D system for testing
        struct SimpleDynamics1D{T} <: AbstractDynamicsModel{1, 0, 1}
            q::T
        end
        
        EstimationToolkit.propagate(m::SimpleDynamics1D, x::SVector, u, dt) = @SVector [x[1] + 0.1 * randn()]
        EstimationToolkit.jacobian_x(m::SimpleDynamics1D, x::SVector, u, dt) = @SMatrix [1.0]
        EstimationToolkit.process_noise_cov(m::SimpleDynamics1D, dt) = @SMatrix [m.q]
        
        struct SimpleMeasurement1D{T} <: AbstractMeasurementModel{1, 1, 1}
            r::T
        end
        
        EstimationToolkit.measure(m::SimpleMeasurement1D, x::SVector) = x
        EstimationToolkit.jacobian_x(m::SimpleMeasurement1D, x::SVector) = @SMatrix [1.0]
        EstimationToolkit.measurement_noise_cov(m::SimpleMeasurement1D) = @SMatrix [m.r]
        
        Random.seed!(42)
        
        x0 = @SVector [0.0]
        P0 = @SMatrix [1.0]
        N_particles = 100
        
        pf = BootstrapParticleFilter(x0, P0, N_particles)
        
        @test num_particles(pf) == N_particles
        @test sum(pf.weights) ≈ 1.0
        
        dyn = SimpleDynamics1D(0.1)
        meas = SimpleMeasurement1D(0.5)
        
        # Run a few steps
        for k in 1:5
            predict!(pf, dyn, nothing, 1.0)
            y = @SVector [0.5]
            update!(pf, meas, y)
        end
        
        @test num_particles(pf) == N_particles
        x_est = state_estimate(pf)
        @test length(x_est) == 1
    end
    
    @testset "Diagnostics" begin
        # Test NIS computation
        μ = @SVector [0.0, 0.0]
        Σ = @SMatrix [1.0 0.0; 0.0 1.0]
        kf = KalmanFilter(μ, Σ)
        
        # Mock innovation statistics
        kf.ν = @SVector [1.0, 1.0]
        kf.S = @SMatrix [1.0 0.0; 0.0 1.0]
        
        nis_val = nis(kf)
        @test nis_val ≈ 2.0  # For this simple case
        
        # Test NEES
        x_true = @SVector [1.0, 1.0]
        x_est = @SVector [0.5, 0.5]
        P = @SMatrix [1.0 0.0; 0.0 1.0]
        
        nees_val = nees(x_true, x_est, P)
        @test nees_val ≈ 0.5  # (0.5^2 + 0.5^2) / 1.0 = 0.5
    end
    
    @testset "Sigma Point Transforms" begin
        # Test unscented transform
        μ = @SVector [1.0, 2.0]
        Σ = @SMatrix [1.0 0.1; 0.1 1.0]
        
        sigma = unscented_transform(μ, Σ)
        
        @test length(sigma.points) == 2 * 2 + 1  # 2n + 1
        @test length(sigma.weights_mean) == 5
        @test length(sigma.weights_cov) == 5
        @test sum(sigma.weights_mean) ≈ 1.0
        
        # Reconstruct should give back original mean and covariance
        μ_recon, Σ_recon = reconstruct_gaussian(sigma.points, sigma.weights_mean, sigma.weights_cov)
        @test μ_recon ≈ μ atol=1e-8  # Relaxed tolerance for numerical precision
        @test Σ_recon ≈ Σ atol=1e-8
        
        # Test cubature transform
        sigma_cub = cubature_transform(μ, Σ)
        @test length(sigma_cub.points) == 2 * 2  # 2n
        @test sum(sigma_cub.weights_mean) ≈ 1.0
    end
    
    @testset "Tier 2: Information Filter" begin
        # Linear system for testing
        μ = @SVector [1.0]  # 1D system
        Σ = @SMatrix [1.0]
        
        inf = InformationFilter(μ, Σ)
        
        # Check conversion to information form
        @test is_positive_definite(inf.Y)
        x_recovered = state_estimate(inf)
        @test x_recovered ≈ μ atol=1e-6
        
        P_recovered = covariance(inf)
        @test P_recovered ≈ Σ atol=1e-6
        
        # Test prediction - use SimpleLinearDynamics
        model = SimpleLinearDynamics(1.0, 0.1)
        u = @SVector zeros(0)
        
        predict!(inf, model, u, 1.0)
        @test is_positive_definite(covariance(inf))
    end
    
    @testset "Tier 2: SRIF" begin
        μ = @SVector [1.0, 0.5]
        Σ = @SMatrix [1.0 0.1; 0.1 1.0]
        
        srif = SquareRootInformationFilter(μ, Σ)
        
        # Check square root is upper triangular (for 2x2, just check lower element)
        @test srif.R̄[2, 1] ≈ 0 atol=1e-10
        
        # Check information matrix is SPD
        Y = information_matrix(srif)
        @test is_positive_definite(Y)
        
        # Check state recovery
        x_recovered = state_estimate(srif)
        @test x_recovered ≈ μ atol=1e-6
        
        # Check covariance recovery
        P_recovered = covariance(srif)
        @test P_recovered ≈ Σ atol=1e-5
    end
    
    @testset "Tier 2: Iterated EKF" begin
        μ = @SVector [1.0]  # 1D system
        Σ = @SMatrix [1.0]
        
        # Create IEKF with 3 max iterations
        iekf = IteratedExtendedKalmanFilter(μ, Σ; max_iterations=3, tolerance=1e-6)
        
        @test iekf.max_iterations == 3
        @test iekf.tolerance == 1e-6
        @test state_estimate(iekf) == μ
        @test covariance(iekf) == Σ
        
        # Test that prediction works - use SimpleLinearDynamics
        model = SimpleLinearDynamics(1.0, 0.1)
        u = @SVector zeros(0)
        
        predict!(iekf, model, u, 1.0)
        @test is_positive_definite(covariance(iekf))
    end
    
    @testset "Tier 2: Cubature Kalman Filter" begin
        μ = @SVector [1.0]  # 1D system
        Σ = @SMatrix [1.0]
        
        ckf = CubatureKalmanFilter(μ, Σ)
        
        @test state_estimate(ckf) == μ
        @test covariance(ckf) == Σ
        
        # Test on linear system - use SimpleLinearDynamics
        model = SimpleLinearDynamics(1.0, 0.1)
        u = @SVector zeros(0)
        
        predict!(ckf, model, u, 1.0)
        
        # Should propagate correctly
        @test is_positive_definite(covariance(ckf))
        
        # For linear system, CKF should give results close to KF
        kf_test = KalmanFilter(μ, Σ)
        predict!(kf_test, model, u, 1.0)
        
        # States should be identical for linear dynamics
        @test state_estimate(ckf) ≈ state_estimate(kf_test) atol=1e-10
    end
    
    @testset "Tier 2: SIR Framework" begin
        # Test bootstrap proposal
        μ = @SVector [1.0, 0.5]
        Σ = @SMatrix [1.0 0.1; 0.1 1.0]
        
        # Create SIR filter with bootstrap proposal
        N_particles = 100
        sir = SequentialImportanceResampling(μ, Σ, N_particles, bootstrap_proposal)
        
        @test num_particles(sir) == N_particles
        @test sum(sir.weights) ≈ 1.0
        
        # Test state estimate is close to initial mean
        x_mean = state_estimate(sir)
        @test norm(x_mean - μ) < 1.0  # Should be reasonably close
        
        # Test effective particles
        ess = effective_particles(sir)
        @test ess > 0
        @test ess ≈ N_particles atol=1e-5  # Should be close to N for uniform weights
    end
    
    @testset "Tier 2: Square-Root Covariance Filter" begin
        # Define simple 1D system
        struct SqrtKF1DDynamics{T} <: AbstractDynamicsModel{1, 0, 1}
            A::T
            Q::T
        end
        
        EstimationToolkit.propagate(m::SqrtKF1DDynamics, x::SVector, u, dt) = @SVector [m.A * x[1]]
        EstimationToolkit.jacobian_x(m::SqrtKF1DDynamics, x::SVector, u, dt) = @SMatrix [m.A]
        EstimationToolkit.process_noise_cov(m::SqrtKF1DDynamics, dt) = @SMatrix [m.Q]
        
        struct SqrtKF1DMeasurement{T} <: AbstractMeasurementModel{1, 1, 1}
            H::T
            R::T
        end
        
        EstimationToolkit.measure(m::SqrtKF1DMeasurement, x::SVector) = @SVector [m.H * x[1]]
        EstimationToolkit.jacobian_x(m::SqrtKF1DMeasurement, x::SVector) = @SMatrix [m.H]
        EstimationToolkit.measurement_noise_cov(m::SqrtKF1DMeasurement) = @SMatrix [m.R]
        
        dyn_sqrt = SqrtKF1DDynamics(0.9, 0.1)
        meas_sqrt = SqrtKF1DMeasurement(1.0, 0.5)
        x0_sqrt = @SVector [1.0]
        P0_sqrt = @SMatrix [1.0]
        
        sqrt_kf = SquareRootCovarianceFilter(x0_sqrt, P0_sqrt)
        
        # Predict
        predict!(sqrt_kf, dyn_sqrt, nothing, 1.0)
        @test sqrt_kf.x[1] ≈ 0.9
        
        # Check that covariance is SPD
        P = covariance(sqrt_kf)
        @test is_positive_definite(P)
        
        # Update
        y = @SVector [0.8]
        update!(sqrt_kf, meas_sqrt, y)
        @test innovation(sqrt_kf) !== nothing
        @test is_positive_definite(covariance(sqrt_kf))
    end
    
    @testset "Tier 2: Quadrature Kalman Filter" begin
        # Test with 2D system for computational feasibility
        x0_2d = @SVector [1.0, 0.5]
        P0_2d = @SMatrix [1.0 0.0; 0.0 1.0]
        
        struct Simple2DDynamics{T} <: AbstractDynamicsModel{2, 0, 2}
            A::SMatrix{2,2,T}
            Q::SMatrix{2,2,T}
        end
        
        EstimationToolkit.propagate(m::Simple2DDynamics, x::SVector, u, dt) = m.A * x
        EstimationToolkit.jacobian_x(m::Simple2DDynamics, x::SVector, u, dt) = m.A
        EstimationToolkit.process_noise_cov(m::Simple2DDynamics, dt) = m.Q
        
        struct Simple2DMeasurement{T} <: AbstractMeasurementModel{2, 2, 2}
            H::SMatrix{2,2,T}
            R::SMatrix{2,2,T}
        end
        
        EstimationToolkit.measure(m::Simple2DMeasurement, x::SVector) = m.H * x
        EstimationToolkit.jacobian_x(m::Simple2DMeasurement, x::SVector) = m.H
        EstimationToolkit.measurement_noise_cov(m::Simple2DMeasurement) = m.R
        
        A = @SMatrix [0.9 0.0; 0.0 0.9]
        Q = @SMatrix [0.1 0.0; 0.0 0.1]
        H = @SMatrix [1.0 0.0; 0.0 1.0]
        R = @SMatrix [0.5 0.0; 0.0 0.5]
        
        dyn2d = Simple2DDynamics(A, Q)
        meas2d = Simple2DMeasurement(H, R)
        
        qkf = QuadratureKalmanFilter(x0_2d, P0_2d, quadrature_order=3)
        
        # Predict
        predict!(qkf, dyn2d, nothing, 1.0)
        # Just check that it produces a valid estimate (not checking exact value due to quadrature)
        @test is_positive_definite(covariance(qkf))
        
        # Update
        y = @SVector [0.8, 0.4]
        update!(qkf, meas2d, y)
        @test innovation(qkf) !== nothing
    end
    
    @testset "Tier 2: Divided Difference Filter" begin
        # Use 2D system
        x0_2d = @SVector [1.0, 0.5]
        P0_2d = @SMatrix [1.0 0.0; 0.0 1.0]
        
        A = @SMatrix [0.9 0.0; 0.0 0.9]
        Q = @SMatrix [0.1 0.0; 0.0 0.1]
        H = @SMatrix [1.0 0.0; 0.0 1.0]
        R = @SMatrix [0.5 0.0; 0.0 0.5]
        
        struct DDF2DDynamics{T} <: AbstractDynamicsModel{2, 0, 2}
            A::SMatrix{2,2,T}
            Q::SMatrix{2,2,T}
        end
        
        EstimationToolkit.propagate(m::DDF2DDynamics, x::SVector, u, dt) = m.A * x
        EstimationToolkit.jacobian_x(m::DDF2DDynamics, x::SVector, u, dt) = m.A
        EstimationToolkit.process_noise_cov(m::DDF2DDynamics, dt) = m.Q
        
        struct DDF2DMeasurement{T} <: AbstractMeasurementModel{2, 2, 2}
            H::SMatrix{2,2,T}
            R::SMatrix{2,2,T}
        end
        
        EstimationToolkit.measure(m::DDF2DMeasurement, x::SVector) = m.H * x
        EstimationToolkit.jacobian_x(m::DDF2DMeasurement, x::SVector) = m.H
        EstimationToolkit.measurement_noise_cov(m::DDF2DMeasurement) = m.R
        
        dyn_ddf = DDF2DDynamics(A, Q)
        meas_ddf = DDF2DMeasurement(H, R)
        
        ddf = DividedDifferenceFilter(x0_2d, P0_2d)
        
        # Predict and update
        predict!(ddf, dyn_ddf, nothing, 1.0)
        @test is_positive_definite(covariance(ddf))
        
        y = @SVector [0.8, 0.4]
        update!(ddf, meas_ddf, y)
        @test innovation(ddf) !== nothing
    end
    
    @testset "Tier 2: Second-Order EKF" begin
        # Use 2D nonlinear system
        x0_2d = @SVector [1.0, 0.5]
        P0_2d = @SMatrix [1.0 0.0; 0.0 1.0]
        
        A = @SMatrix [0.9 0.0; 0.0 0.9]
        Q = @SMatrix [0.1 0.0; 0.0 0.1]
        H = @SMatrix [1.0 0.0; 0.0 1.0]
        R = @SMatrix [0.5 0.0; 0.0 0.5]
        
        struct SOEKF2DDynamics{T} <: AbstractDynamicsModel{2, 0, 2}
            A::SMatrix{2,2,T}
            Q::SMatrix{2,2,T}
        end
        
        EstimationToolkit.propagate(m::SOEKF2DDynamics, x::SVector, u, dt) = m.A * x
        EstimationToolkit.jacobian_x(m::SOEKF2DDynamics, x::SVector, u, dt) = m.A
        EstimationToolkit.process_noise_cov(m::SOEKF2DDynamics, dt) = m.Q
        
        struct SOEKF2DMeasurement{T} <: AbstractMeasurementModel{2, 2, 2}
            H::SMatrix{2,2,T}
            R::SMatrix{2,2,T}
        end
        
        EstimationToolkit.measure(m::SOEKF2DMeasurement, x::SVector) = m.H * x
        EstimationToolkit.jacobian_x(m::SOEKF2DMeasurement, x::SVector) = m.H
        EstimationToolkit.measurement_noise_cov(m::SOEKF2DMeasurement) = m.R
        
        dyn_soekf = SOEKF2DDynamics(A, Q)
        meas_soekf = SOEKF2DMeasurement(H, R)
        
        soekf = SecondOrderExtendedKalmanFilter(x0_2d, P0_2d)
        
        # Predict and update
        predict!(soekf, dyn_soekf, nothing, 1.0)
        @test is_positive_definite(covariance(soekf))
        
        y = @SVector [0.8, 0.4]
        update!(soekf, meas_soekf, y)
        @test innovation(soekf) !== nothing
    end
    
    @testset "Tier 2: Auxiliary Particle Filter" begin
        # Define simple 1D system
        struct APF1DDynamics{T} <: AbstractDynamicsModel{1, 0, 1}
            A::T
            Q::T
        end
        
        EstimationToolkit.propagate(m::APF1DDynamics, x::SVector, u, dt) = @SVector [m.A * x[1]]
        EstimationToolkit.jacobian_x(m::APF1DDynamics, x::SVector, u, dt) = @SMatrix [m.A]
        EstimationToolkit.process_noise_cov(m::APF1DDynamics, dt) = @SMatrix [m.Q]
        
        struct APF1DMeasurement{T} <: AbstractMeasurementModel{1, 1, 1}
            H::T
            R::T
        end
        
        EstimationToolkit.measure(m::APF1DMeasurement, x::SVector) = @SVector [m.H * x[1]]
        EstimationToolkit.jacobian_x(m::APF1DMeasurement, x::SVector) = @SMatrix [m.H]
        EstimationToolkit.measurement_noise_cov(m::APF1DMeasurement) = @SMatrix [m.R]
        
        dyn_apf = APF1DDynamics(0.9, 0.1)
        meas_apf = APF1DMeasurement(1.0, 0.5)
        
        N_particles = 100
        μ = @SVector [1.0]
        Σ = @SMatrix [1.0]
        
        apf = AuxiliaryParticleFilter(μ, Σ, N_particles)
        
        @test num_particles(apf) == N_particles
        @test sum(apf.weights) ≈ 1.0
        
        # Test predict and update
        predict!(apf, dyn_apf, nothing, 1.0)
        @test num_particles(apf) == N_particles
        
        y = @SVector [0.8]
        update!(apf, meas_apf, y)
        @test sum(apf.weights) ≈ 1.0
        
        # Test state estimate
        x_mean = state_estimate(apf)
        @test length(x_mean) == 1
    end
    
    @testset "Tier 2: Regularized Particle Filter" begin
        # Define simple 1D system
        struct RPF1DDynamics{T} <: AbstractDynamicsModel{1, 0, 1}
            A::T
            Q::T
        end
        
        EstimationToolkit.propagate(m::RPF1DDynamics, x::SVector, u, dt) = @SVector [m.A * x[1]]
        EstimationToolkit.jacobian_x(m::RPF1DDynamics, x::SVector, u, dt) = @SMatrix [m.A]
        EstimationToolkit.process_noise_cov(m::RPF1DDynamics, dt) = @SMatrix [m.Q]
        
        struct RPF1DMeasurement{T} <: AbstractMeasurementModel{1, 1, 1}
            H::T
            R::T
        end
        
        EstimationToolkit.measure(m::RPF1DMeasurement, x::SVector) = @SVector [m.H * x[1]]
        EstimationToolkit.jacobian_x(m::RPF1DMeasurement, x::SVector) = @SMatrix [m.H]
        EstimationToolkit.measurement_noise_cov(m::RPF1DMeasurement) = @SMatrix [m.R]
        
        dyn_rpf = RPF1DDynamics(0.9, 0.1)
        meas_rpf = RPF1DMeasurement(1.0, 0.5)
        
        N_particles = 100
        μ = @SVector [1.0]
        Σ = @SMatrix [1.0]
        
        rpf = RegularizedParticleFilter(μ, Σ, N_particles)
        
        @test num_particles(rpf) == N_particles
        @test sum(rpf.weights) ≈ 1.0
        
        # Test predict and update
        predict!(rpf, dyn_rpf, nothing, 1.0)
        @test num_particles(rpf) == N_particles
        
        y = @SVector [0.8]
        update!(rpf, meas_rpf, y)
        @test sum(rpf.weights) ≈ 1.0
        
        # Test that regularization maintains diversity
        ess_before = effective_particles(rpf)
        @test ess_before > 0
        
        # Multiple updates to trigger resampling
        for i in 1:5
            y_test = @SVector [0.8 + 0.1*randn()]
            update!(rpf, meas_rpf, y_test)
        end
        
        ess_after = effective_particles(rpf)
        @test ess_after > N_particles * 0.1  # Should maintain some diversity
    end
    
end

# Include new test suites for recently added filters
include("test_smoothers.jl")
include("test_grid_filters.jl")
include("test_multiple_model.jl")
include("test_multiple_model_consistency.jl")
include("test_advanced_particle.jl")
include("test_flow_ensemble.jl")

println("\n✓ All tests passed!")
