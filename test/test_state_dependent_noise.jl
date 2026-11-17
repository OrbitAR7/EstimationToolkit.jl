"""
State-dependent process noise: `process_noise_cov(model, x, dt)`.

A model may define the three-argument form; the Kalman-type filters evaluate it at
the estimate before propagation. Models that define only `process_noise_cov(model, dt)`
are unaffected.
"""

using Test
using EstimationToolkit
using LinearAlgebra
using StaticArrays

@testset "State-dependent process noise" begin

    # Scalar: x⁺ = a·x, Q(x) = (σ·x)²·dt. Only the three-argument form is defined.
    struct SDScalar <: AbstractDynamicsModel{1, 0, 1}
        a::Float64
        σ::Float64
    end
    EstimationToolkit.propagate(m::SDScalar, x, u, dt) = SVector(m.a * x[1])
    EstimationToolkit.jacobian_x(m::SDScalar, x::SVector, u, dt) = SMatrix{1,1}(m.a)
    EstimationToolkit.process_noise_cov(m::SDScalar, x, dt) = SMatrix{1,1}((m.σ * x[1])^2 * dt)

    # Two states, linear dynamics [h, v]: h⁺ = h + v·dt, v⁺ = v. The noise on v
    # scales with v (the descent-rate case that motivated the interface).
    struct SDDescent <: AbstractDynamicsModel{2, 0, 2}
        q_h::Float64
        σ_v::Float64
    end
    EstimationToolkit.propagate(m::SDDescent, x, u, dt) = SVector(x[1] + x[2] * dt, x[2])
    EstimationToolkit.jacobian_x(m::SDDescent, x::SVector, u, dt) = @SMatrix [1.0 dt; 0.0 1.0]
    EstimationToolkit.process_noise_cov(m::SDDescent, x, dt) =
        @SMatrix [m.q_h*dt 0.0; 0.0 (m.σ_v * x[2])^2*dt]

    # Nonlinear dynamics with the same noise, to pin down the evaluation point.
    struct SDNonlinear <: AbstractDynamicsModel{2, 0, 2}
        σ_v::Float64
    end
    EstimationToolkit.propagate(m::SDNonlinear, x, u, dt) =
        SVector(x[1] + x[2] * dt, x[2] - 0.1 * x[2]^2 * dt)
    EstimationToolkit.jacobian_x(m::SDNonlinear, x::SVector, u, dt) =
        @SMatrix [1.0 dt; 0.0 1.0 - 0.2*x[2]*dt]
    EstimationToolkit.process_noise_cov(m::SDNonlinear, x, dt) =
        @SMatrix [1e-4*dt 0.0; 0.0 (m.σ_v * x[2])^2*dt]

    # State-independent model: only the two-argument form.
    struct SDConstant <: AbstractDynamicsModel{2, 0, 2}
        Q::SMatrix{2,2,Float64,4}
    end
    EstimationToolkit.propagate(m::SDConstant, x, u, dt) = SVector(x[1] + x[2] * dt, x[2])
    EstimationToolkit.jacobian_x(m::SDConstant, x::SVector, u, dt) = @SMatrix [1.0 dt; 0.0 1.0]
    EstimationToolkit.process_noise_cov(m::SDConstant, dt) = m.Q

    u = Float64[]

    # Every filter whose predict! evaluates Q at the estimate.
    make_filters(x0, P0) = [
        "KalmanFilter" => KalmanFilter(x0, P0),
        "InformationFilter" => InformationFilter(x0, P0),
        "SquareRootInformationFilter" => SquareRootInformationFilter(x0, P0),
        "SquareRootCovarianceFilter" => SquareRootCovarianceFilter(x0, P0),
        "FadingMemoryFilter (α = 1)" => FadingMemoryFilter(x0, P0; α = 1.0),
        "ExtendedKalmanFilter" => ExtendedKalmanFilter(x0, P0),
        "IteratedExtendedKalmanFilter" => IteratedExtendedKalmanFilter(x0, P0),
        "SecondOrderExtendedKalmanFilter" => SecondOrderExtendedKalmanFilter(x0, P0),
        "UnscentedKalmanFilter" => UnscentedKalmanFilter(x0, P0),
        "CubatureKalmanFilter" => CubatureKalmanFilter(x0, P0),
        "QuadratureKalmanFilter" => QuadratureKalmanFilter(x0, P0),
        "DividedDifferenceFilter" => DividedDifferenceFilter(x0, P0),
    ]

    @testset "Hand-computed scalar step" begin
        # x = 2, P = 0.5, a = 0.9, σ = 0.1, dt = 1:
        #   Q(x) = (0.1·2)² = 0.04,  P̄ = 0.9²·0.5 + 0.04 = 0.445,  x̄ = 1.8
        model = SDScalar(0.9, 0.1)
        x0 = @SVector [2.0]
        P0 = @SMatrix [0.5]
        @test process_noise_cov(model, x0, 1.0)[1, 1] ≈ 0.04

        for (name, f) in make_filters(x0, P0)
            @testset "$name" begin
                predict!(f, model, u, 1.0)
                @test state_estimate(f)[1] ≈ 1.8 atol = 1e-9
                @test covariance(f)[1, 1] ≈ 0.445 atol = 1e-9
            end
        end

        # Second step: Q now follows the propagated estimate, x = 1.8.
        #   Q = (0.1·1.8)² = 0.0324,  P̄ = 0.81·0.445 + 0.0324 = 0.39285
        for F in (ExtendedKalmanFilter, UnscentedKalmanFilter)
            f = F(x0, P0)
            predict!(f, model, u, 1.0)
            predict!(f, model, u, 1.0)
            @test state_estimate(f)[1] ≈ 1.62 atol = 1e-9
            @test covariance(f)[1, 1] ≈ 0.39285 atol = 1e-9
        end
    end

    @testset "Two states, linear dynamics" begin
        model = SDDescent(1e-3, 0.05)
        dt = 0.5
        x0 = @SVector [1000.0, -40.0]
        P0 = @SMatrix [25.0 1.5; 1.5 4.0]
        F = @SMatrix [1.0 dt; 0.0 1.0]
        Q = @SMatrix [1e-3*dt 0.0; 0.0 (0.05 * 40.0)^2*dt]      # diag(5e-4, 2.0)
        P_expected = F * P0 * F' + Q
        @test Q[2, 2] ≈ 2.0

        for (name, f) in make_filters(x0, P0)
            @testset "$name" begin
                predict!(f, model, u, dt)
                # Relative tolerance: the sigma-point filters reconstruct the mean
                # from points ~√N·σ apart, which costs a few ulps at 1000.
                @test state_estimate(f) ≈ [980.0, -40.0] rtol = 1e-10
                @test covariance(f) ≈ P_expected atol = 1e-8
            end
        end
    end

    @testset "Q is evaluated before propagation" begin
        # Nonlinear dynamics change v over the step, so Q at the prior estimate and
        # Q at the propagated estimate differ. The EKF must use the prior one.
        model = SDNonlinear(0.05)
        dt = 0.5
        x0 = @SVector [1000.0, -40.0]
        P0 = @SMatrix [25.0 1.5; 1.5 4.0]
        F = jacobian_x(model, x0, u, dt)
        x1 = propagate(model, x0, u, dt)
        Q_prior = process_noise_cov(model, x0, dt)
        Q_post = process_noise_cov(model, x1, dt)
        @test !(Q_prior ≈ Q_post)

        ekf = ExtendedKalmanFilter(x0, P0)
        predict!(ekf, model, u, dt)
        @test ekf.x ≈ x1
        @test ekf.P ≈ F * P0 * F' + Q_prior
        @test !(ekf.P ≈ F * P0 * F' + Q_post)

        # The UKF adds the same Q to its sigma-point covariance.
        ukf = UnscentedKalmanFilter(x0, P0)
        sigma = unscented_transform(x0, P0; α = ukf.α, β = ukf.β, κ = ukf.κ)
        χ = transform_sigma_points(s -> propagate(model, s, u, dt), sigma)
        _, P_sigma = reconstruct_gaussian(χ, sigma.weights_mean, sigma.weights_cov)
        predict!(ukf, model, u, dt)
        @test ukf.P ≈ make_symmetric(P_sigma + Q_prior)
    end

    @testset "State-independent models are unchanged" begin
        Q = @SMatrix [0.02 0.001; 0.001 0.3]
        model = SDConstant(Q)
        dt = 0.5
        x0 = @SVector [1000.0, -40.0]
        P0 = @SMatrix [25.0 1.5; 1.5 4.0]
        F = @SMatrix [1.0 dt; 0.0 1.0]

        # The three-argument form falls back to the two-argument one, for any state.
        @test process_noise_cov(model, x0, dt) === process_noise_cov(model, dt)
        @test process_noise_cov(model, [1.0, 2.0], dt) === Q
        @test process_noise_cov(model, nothing, dt) === Q

        for (name, f) in make_filters(x0, P0)
            @testset "$name" begin
                predict!(f, model, u, dt)
                @test covariance(f) ≈ F * P0 * F' + Q atol = 1e-8
            end
        end
    end

    @testset "Filters built on the EKF pick it up" begin
        # A one-component Gaussian sum filter is an EKF.
        model = SDDescent(1e-3, 0.05)
        dt = 0.5
        x0 = @SVector [1000.0, -40.0]
        P0 = @SMatrix [25.0 1.5; 1.5 4.0]
        ekf = ExtendedKalmanFilter(x0, P0)
        gsf = GaussianSumFilter([ExtendedKalmanFilter(x0, P0)], [1.0])
        predict!(ekf, model, u, dt)
        predict!(gsf, model, u, dt)
        @test covariance(gsf) ≈ ekf.P
    end

    # A position-only measurement of the two-state model: fewer measurements than
    # states, a non-diagonal prior, so the transposes in a square-root filter matter.
    struct SDPositionMeasurement <: AbstractMeasurementModel{2, 1, 1}
        σ::Float64
    end
    EstimationToolkit.measure(m::SDPositionMeasurement, x::SVector) = SVector(x[1])
    EstimationToolkit.jacobian_x(m::SDPositionMeasurement, x::SVector) = @SMatrix [1.0 0.0]
    EstimationToolkit.measurement_noise_cov(m::SDPositionMeasurement) = SMatrix{1,1}(m.σ^2)

    @testset "SquareRootCovarianceFilter matches the Kalman filter" begin
        # Regression test: predict! and update! used S·F' and S·(I-KH)' where the
        # stacked QR needs S'·F' and S'·(I-KH)'. That is only the same when S is
        # symmetric, so the 1-D test in runtests.jl never caught it; update! also
        # failed outright when the measurement had fewer components than the state.
        dyn = SDDescent(1e-3, 0.05)
        meas = SDPositionMeasurement(3.0)
        dt = 0.5
        x0 = @SVector [1000.0, -40.0]
        P0 = @SMatrix [25.0 1.5; 1.5 4.0]
        kf = KalmanFilter(x0, P0)
        srcf = SquareRootCovarianceFilter(x0, P0)
        for y in (978.0, 962.5, 939.0, 921.0)
            predict!(kf, dyn, u, dt)
            predict!(srcf, dyn, u, dt)
            @test state_estimate(srcf) ≈ state_estimate(kf) rtol = 1e-10
            @test covariance(srcf) ≈ covariance(kf) rtol = 1e-9
            update!(kf, meas, SVector(y))
            update!(srcf, meas, SVector(y))
            @test state_estimate(srcf) ≈ state_estimate(kf) rtol = 1e-10
            @test covariance(srcf) ≈ covariance(kf) rtol = 1e-9
        end
    end
end
