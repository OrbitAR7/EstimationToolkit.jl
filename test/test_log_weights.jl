using Test
using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random

# Weight updates must survive measurements that are far in the tail of every
# sample. With `w *= exp(loglik)` all weights become exactly zero and the
# measurement is thrown away; in log space the ratios are kept.

@testset "Log-space weight updates" begin

    struct LWRandomWalk <: AbstractDynamicsModel{1, 0, 1}
        q::Float64
    end
    EstimationToolkit.propagate(::LWRandomWalk, x, u, dt) = x
    EstimationToolkit.process_noise_cov(m::LWRandomWalk, dt) = SMatrix{1,1}(m.q * dt)

    struct LWDirect <: AbstractMeasurementModel{1, 1, 1}
        σ::Float64
    end
    EstimationToolkit.measure(::LWDirect, x) = SVector(x[1])
    EstimationToolkit.measurement_noise_cov(m::LWDirect) = SMatrix{1,1}(m.σ^2)

    @testset "reweight_log!" begin
        # Agrees with the direct product when nothing underflows
        w = [0.2, 0.3, 0.5]
        ll = [-1.0, -2.0, -0.5]
        expected = w .* exp.(ll)
        expected ./= sum(expected)
        @test EstimationToolkit.reweight_log!(w, ll) == true
        @test w ≈ expected
        @test sum(w) ≈ 1.0

        # A common offset in the log-likelihoods changes nothing
        w1 = [0.2, 0.3, 0.5]
        w2 = [0.2, 0.3, 0.5]
        EstimationToolkit.reweight_log!(w1, ll)
        EstimationToolkit.reweight_log!(w2, ll .- 1.0e6)
        @test w1 ≈ w2

        # Every exp() underflows (exp(-800) == 0.0), ratios must survive
        @test exp(-800.0) == 0.0
        w = [0.5, 0.5]
        EstimationToolkit.reweight_log!(w, [-800.0, -810.0])
        @test w[1] ≈ 1 / (1 + exp(-10.0))
        @test w[2] ≈ exp(-10.0) / (1 + exp(-10.0))

        # A zero prior weight stays zero; NaN is treated as impossible
        w = [0.0, 1.0]
        EstimationToolkit.reweight_log!(w, [0.0, -900.0])
        @test w == [0.0, 1.0]
        w = [0.5, 0.5]
        EstimationToolkit.reweight_log!(w, [NaN, -3.0])
        @test w == [0.0, 1.0]

        # Nothing finite: weights untouched, caller is told
        w = [0.5, 0.5]
        @test EstimationToolkit.reweight_log!(w, [-Inf, -Inf]) == false
        @test w == [0.5, 0.5]
    end

    @testset "log_likelihood matches likelihood" begin
        meas = LWDirect(0.5)
        x = SVector(1.0)
        y = SVector(1.3)
        @test exp(EstimationToolkit.log_likelihood(meas, y, x)) ≈
              EstimationToolkit.likelihood(meas, y, x)
        # closed form: N(1.3; 1.0, 0.25)
        @test EstimationToolkit.likelihood(meas, y, x) ≈
              exp(-0.5 * 0.3^2 / 0.25) / sqrt(2π * 0.25)
        # far tail: likelihood underflows, the log does not
        y_far = SVector(1.0e4)
        @test EstimationToolkit.likelihood(meas, y_far, x) == 0.0
        @test isfinite(EstimationToolkit.log_likelihood(meas, y_far, x))
    end

    @testset "Gaussian sum filter" begin
        dyn = LWRandomWalk(0.01)
        P0 = SMatrix{1,1}(1.0)

        @testset "one component reproduces the EKF" begin
            meas = LWDirect(0.5)
            ekf = ExtendedKalmanFilter(SVector(0.3), P0)
            gsf = GaussianSumFilter([ExtendedKalmanFilter(SVector(0.3), P0)], [1.0])
            rng = MersenneTwister(1)
            for _ in 1:20
                y = SVector(1.0 + 0.5 * randn(rng))
                predict!(ekf, dyn, nothing, 1.0); update!(ekf, meas, y)
                predict!(gsf, dyn, nothing, 1.0); update!(gsf, meas, y)
            end
            @test state_estimate(gsf) ≈ state_estimate(ekf)
            @test covariance(gsf) ≈ covariance(ekf)
            @test innovation(gsf) ≈ innovation(ekf)
        end

        @testset "weights follow Bayes' rule" begin
            # Two components, linear measurement: posterior weights are
            # w_i N(y; μ_i, P_i + R) normalized.
            meas = LWDirect(0.5)
            μ = (-1.0, 2.0)
            gsf = GaussianSumFilter([ExtendedKalmanFilter(SVector(μ[1]), P0),
                                     ExtendedKalmanFilter(SVector(μ[2]), P0)], [0.3, 0.7])
            y = 0.2
            update!(gsf, meas, SVector(y))
            S = 1.0 + 0.25
            L = [0.3 * exp(-0.5 * (y - μ[1])^2 / S), 0.7 * exp(-0.5 * (y - μ[2])^2 / S)]
            @test gsf.weights ≈ L ./ sum(L)
        end

        @testset "tight measurement far from every component" begin
            # σ = 1e-3 and components 40 and 60 units away from the measurement
            # after accounting for the prior: both likelihoods are exactly 0.0
            # in floating point, but the closer component must take the weight.
            meas = LWDirect(1e-3)
            Ptight = SMatrix{1,1}(1e-6)
            gsf = GaussianSumFilter([ExtendedKalmanFilter(SVector(0.0), Ptight),
                                     ExtendedKalmanFilter(SVector(10.0), Ptight)],
                                    [0.5, 0.5]; weight_threshold = 1e-300)
            y = 4.0   # 4 from the first, 6 from the second
            S = 1e-6 + 1e-6
            @test exp(-0.5 * (y - 0.0)^2 / S) == 0.0     # the old update saw only zeros
            @test exp(-0.5 * (y - 10.0)^2 / S) == 0.0
            update!(gsf, meas, SVector(y))
            @test num_components(gsf) >= 1
            @test all(isfinite, gsf.weights)
            @test sum(gsf.weights) ≈ 1.0
            # the surviving / dominant component is the one that started at 0
            i = argmax(gsf.weights)
            @test gsf.weights[i] > 0.999999
            @test abs(state_estimate(gsf.components[i])[1] - 2.0) < 1e-6  # midway: equal P and R
        end
    end

    @testset "Particle filters keep the measurement" begin
        # Particles on a grid over [-50, 50] (spacing 0.5); measurement at 20.2
        # with σ = 1e-3. The nearest particle (20.0) is 200σ away, so exp()
        # underflows for every one of them. The posterior must still concentrate
        # on that nearest particle.
        N = 201
        grid = collect(range(-50.0, 50.0; length = N))
        meas = LWDirect(1e-3)
        y = SVector(20.2)
        @test all(x -> exp(-0.5 * (y[1] - x)^2 / 1e-6) == 0.0, grid)
        make() = [SVector(x) for x in grid]
        w0() = fill(1.0 / N, N)

        @testset "bootstrap" begin
            pf = BootstrapParticleFilter(make(), w0(); rng = MersenneTwister(2))
            update!(pf, meas, y)
            @test abs(state_estimate(pf)[1] - 20.0) < 1e-6
        end

        # These two only have a (mean, covariance, N) constructor, so build one
        # and then place the particles on the grid.
        x0, P0 = SVector(0.0), SMatrix{1,1}(1.0)

        @testset "regularized" begin
            Random.seed!(3)
            rpf = RegularizedParticleFilter(x0, P0, N)
            rpf.particles = make(); rpf.weights = w0()
            update!(rpf, meas, y)
            # All the weight lands on one particle, so the kernel bandwidth is
            # zero and the filter must fall back to plain resampling, not throw.
            @test abs(state_estimate(rpf)[1] - 20.0) < 1e-6
            @test all(isfinite, rpf.weights)
        end

        @testset "auxiliary" begin
            Random.seed!(4)
            apf = AuxiliaryParticleFilter(x0, P0, N)
            apf.particles = make(); apf.weights = w0()
            update!(apf, meas, y)
            @test abs(state_estimate(apf)[1] - 20.0) < 1e-6
        end
    end
end
