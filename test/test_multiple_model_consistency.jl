"""
Checks that the multiple-model filters reduce to known cases and handle
large innovations:

- with a single mode, IMM / GPB1 / GPB2 must reproduce a plain Kalman filter
- GPB2 with M identical models must also reproduce it
- a jump of hundreds of sigma must move the mode probabilities (this used to
  freeze them, because every likelihood underflowed to zero)
- the EKF works inside the filter banks
"""

using Test
using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random

struct MMCV <: AbstractDynamicsModel{2, 0, 2}
    q::Float64
end
EstimationToolkit.propagate(m::MMCV, x::SVector, u, dt) = @SVector [x[1] + x[2] * dt, x[2]]
EstimationToolkit.jacobian_x(m::MMCV, x::SVector, u, dt) = @SMatrix [1.0 dt; 0.0 1.0]
EstimationToolkit.process_noise_cov(m::MMCV, dt) =
    m.q * @SMatrix [dt^3/3 dt^2/2; dt^2/2 dt]

struct MMPos <: AbstractMeasurementModel{2, 1, 1}
    σ::Float64
end
EstimationToolkit.measure(m::MMPos, x::SVector) = @SVector [x[1]]
EstimationToolkit.jacobian_x(m::MMPos, x::SVector) = @SMatrix [1.0 0.0]
EstimationToolkit.measurement_noise_cov(m::MMPos) = @SMatrix [m.σ^2]

@testset "Multiple-model consistency" begin
    x0 = @SVector [0.0, 1.0]
    P0 = @SMatrix [4.0 0.0; 0.0 1.0]
    dyn = MMCV(0.01)
    meas = MMPos(0.5)
    u = zeros(0)
    rng = MersenneTwister(7)
    ys = [@SVector([k * 1.0 + 0.5 * randn(rng)]) for k in 1:30]

    kf = KalmanFilter(x0, P0)
    for y in ys
        predict!(kf, dyn, u, 1.0)
        update!(kf, meas, y)
    end

    @testset "single mode reduces to a Kalman filter" begin
        Π1 = fill(1.0, 1, 1)
        imm = InteractingMultipleModel([KalmanFilter(x0, P0)], Π1, [1.0])
        gpb1 = GPB1Filter([KalmanFilter(x0, P0)], Π1, [1.0])
        gpb2 = GPB2Filter(KalmanFilter(x0, P0), 1, Π1, [1.0])
        for y in ys
            step!(imm, [dyn], [meas], u, y, 1.0)
            step!(gpb1, [dyn], [meas], u, y, 1.0)
            step!(gpb2, [dyn], [meas], u, y, 1.0)
        end
        for f in (imm, gpb1, gpb2)
            @test state_estimate(f) ≈ state_estimate(kf) atol=1e-10
            @test covariance(f) ≈ covariance(kf) atol=1e-10
        end
    end

    @testset "GPB2 with identical models reduces to a Kalman filter" begin
        Π = [0.9 0.1; 0.2 0.8]
        gpb2 = GPB2Filter(KalmanFilter(x0, P0), 2, Π, [0.5, 0.5])
        for y in ys
            step!(gpb2, [dyn, dyn], [meas, meas], u, y, 1.0)
        end
        @test state_estimate(gpb2) ≈ state_estimate(kf) atol=1e-10
        @test covariance(gpb2) ≈ covariance(kf) atol=1e-10
        # identical likelihoods: mode probabilities follow the Markov chain alone
        @test sum(gpb2.mode_probs) ≈ 1.0
    end

    @testset "GPB1 restarts every filter from the merged estimate" begin
        Π = [0.95 0.05; 0.05 0.95]
        gpb1 = GPB1Filter([KalmanFilter(x0, P0), KalmanFilter(x0, P0)], Π, [0.5, 0.5])
        quiet, noisy = MMCV(0.01), MMCV(10.0)
        for y in ys
            step!(gpb1, [quiet, noisy], [meas, meas], u, y, 1.0)
        end
        # Both filters were started from the same merged state one step ago and
        # differ only through one predict/update, so they must stay close.
        d = state_estimate(gpb1.filters[1]) - state_estimate(gpb1.filters[2])
        @test norm(d) < 1.0
    end

    @testset "large innovations still move the mode probabilities" begin
        Π = [0.98 0.02; 0.10 0.90]
        quiet, maneuver = MMCV(1e-6), MMCV(1e4)
        tight = MMPos(0.01)
        for make in (
            () -> InteractingMultipleModel([KalmanFilter(x0, P0), KalmanFilter(x0, P0)], Π, [0.9, 0.1]),
            () -> GPB1Filter([KalmanFilter(x0, P0), KalmanFilter(x0, P0)], Π, [0.9, 0.1]),
            () -> GPB2Filter(KalmanFilter(x0, P0), 2, Π, [0.9, 0.1]),
        )
            f = make()
            for k in 1:20
                step!(f, [quiet, maneuver], [tight, tight], u, @SVector([k * 1.0]), 1.0)
            end
            @test get_mode_probabilities(f)[1] > 0.9
            # a 50 m jump with 1 cm noise is a 5000-sigma innovation
            step!(f, [quiet, maneuver], [tight, tight], u, @SVector([21.0 + 50.0]), 1.0)
            @test get_mode_probabilities(f)[2] > 0.99
            @test all(isfinite, state_estimate(f))
        end
    end

    @testset "EKF inside the filter banks" begin
        Π = [0.95 0.05; 0.05 0.95]
        imm = InteractingMultipleModel([ExtendedKalmanFilter(x0, P0), ExtendedKalmanFilter(x0, P0)], Π)
        gpb2 = GPB2Filter(ExtendedKalmanFilter(x0, P0), 2, Π)
        for y in ys
            step!(imm, [dyn, MMCV(1.0)], [meas, meas], u, y, 1.0)
            step!(gpb2, [dyn, MMCV(1.0)], [meas, meas], u, y, 1.0)
        end
        @test abs(state_estimate(imm)[1] - 30.0) < 2.0
        @test abs(state_estimate(gpb2)[1] - 30.0) < 2.0
    end
end
