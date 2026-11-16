using Test
using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random
using Statistics

# Regression tests for bugs that only showed with a measurement smaller than the
# state, a state larger than 3, or a singular process-noise covariance. All three
# are the normal case in orbit determination (6 states, angles-only measurements,
# noise entering through the accelerations).

@testset "Non-square measurements, 6 states, singular Q" begin

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    # 3-D constant velocity, 6 states
    struct NSConstVel <: AbstractDynamicsModel{6, 0, 6}
        q::Float64
    end
    EstimationToolkit.propagate(::NSConstVel, x, u, dt) =
        SVector(x[1] + dt * x[4], x[2] + dt * x[5], x[3] + dt * x[6], x[4], x[5], x[6])
    function EstimationToolkit.process_noise_cov(m::NSConstVel, dt)
        I3 = SMatrix{3,3}(1.0I)
        return m.q * [dt^3/3 * I3 dt^2/2 * I3; dt^2/2 * I3 dt * I3]
    end

    # Same dynamics, process noise only through the accelerations: rank 3
    struct NSAccelNoise <: AbstractDynamicsModel{6, 0, 6}
        σ::Float64
    end
    EstimationToolkit.propagate(::NSAccelNoise, x, u, dt) =
        EstimationToolkit.propagate(NSConstVel(0.0), x, u, dt)
    function EstimationToolkit.process_noise_cov(m::NSAccelNoise, dt)
        G = vcat(0.5dt^2 * SMatrix{3,3}(1.0I), dt * SMatrix{3,3}(1.0I))
        return m.σ^2 * (G * G')
    end

    # Two of six components measured
    struct NSPosXY <: AbstractMeasurementModel{6, 2, 2}
        σ::Float64
    end
    EstimationToolkit.measure(::NSPosXY, x) = SVector(x[1], x[2])
    EstimationToolkit.measurement_noise_cov(m::NSPosXY) = SMatrix{2,2}(m.σ^2 * I)

    # Quadratic models for the second-order EKF (its moments are exact for these)
    struct NSQuadMeas <: AbstractMeasurementModel{2, 1, 1}
        σ::Float64
    end
    EstimationToolkit.measure(::NSQuadMeas, x) = SVector(x[1]^2 + 0.5 * x[1] * x[2])
    EstimationToolkit.measurement_noise_cov(m::NSQuadMeas) = SMatrix{1,1}(m.σ^2)

    struct NSQuadDyn <: AbstractDynamicsModel{2, 0, 2} end
    EstimationToolkit.propagate(::NSQuadDyn, x, u, dt) =
        SVector(x[1] + 0.2 * x[2]^2, x[2] + 0.1 * x[1] * x[2])
    EstimationToolkit.process_noise_cov(::NSQuadDyn, dt) = SMatrix{2,2}(1e-8I)

    # ------------------------------------------------------------------
    # 1. Linear 6-state system, 2-D measurement: every Gaussian filter is the
    #    Kalman filter
    # ------------------------------------------------------------------
    @testset "Linear system: filters reproduce the Kalman filter" begin
        dt = 1.0
        dyn = NSConstVel(0.01)
        meas = NSPosXY(0.5)
        I3 = Matrix(1.0I, 3, 3)
        F = [I3 dt * I3; zeros(3, 3) I3]
        H = [1.0 0 0 0 0 0; 0 1.0 0 0 0 0]
        Q = Matrix(process_noise_cov(dyn, dt))
        R = Matrix(measurement_noise_cov(meas))

        rng = MersenneTwister(11)
        x0 = SVector{6}(randn(rng, 6))
        A = randn(rng, 6, 6)
        P0 = SMatrix{6,6}(A * A' + 6I)
        ys = [SVector{2}(randn(rng, 2)) for _ in 1:10]

        # reference Kalman filter
        xk, Pk = Vector(x0), Matrix(P0)
        ref = Tuple{Vector{Float64}, Matrix{Float64}}[]
        for y in ys
            xk = F * xk
            Pk = F * Pk * F' + Q
            S = H * Pk * H' + R
            K = Pk * H' / S
            xk = xk + K * (Vector(y) - H * xk)
            Pk = Pk - K * S * K'
            push!(ref, (copy(xk), copy(Pk)))
        end

        filters = [
            ("SecondOrderEKF", () -> SecondOrderExtendedKalmanFilter(x0, P0)),
            ("QuadratureKF order 3", () -> QuadratureKalmanFilter(x0, P0; quadrature_order=3)),
            ("QuadratureKF order 2", () -> QuadratureKalmanFilter(x0, P0; quadrature_order=2)),
            ("DividedDifference", () -> DividedDifferenceFilter(x0, P0)),
            ("DividedDifference h=2", () -> DividedDifferenceFilter(x0, P0; h=2.0)),
            ("EKF", () -> ExtendedKalmanFilter(x0, P0)),
            ("Cubature", () -> CubatureKalmanFilter(x0, P0)),
        ]
        for (name, make) in filters
            @testset "$name" begin
                f = make()
                for (k, y) in enumerate(ys)
                    predict!(f, dyn, nothing, dt)
                    update!(f, meas, y)
                    @test state_estimate(f) ≈ ref[k][1] rtol = 1e-8 atol = 1e-10
                    @test Matrix(covariance(f)) ≈ ref[k][2] rtol = 1e-8 atol = 1e-10
                end
            end
        end
    end

    @testset "Divided difference: predicted covariance in 6 states" begin
        # earlier versions returned (h^2/n) * (F P F'), half of it for n = 6
        dyn = NSConstVel(0.0)
        x0 = @SVector zeros(6)
        P0 = SMatrix{6,6}(Diagonal([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]))
        f = DividedDifferenceFilter(x0, P0)
        predict!(f, dyn, nothing, 2.0)
        F = Matrix(EstimationToolkit.jacobian_x(dyn, x0, nothing, 2.0))
        @test Matrix(covariance(f)) ≈ F * Matrix(P0) * F' rtol = 1e-12
        @test_throws ArgumentError DividedDifferenceFilter(x0, P0; h=0.5)
    end

    @testset "Gauss-Hermite rule: standard-normal moments" begin
        for n in (1, 2, 3, 5, 8)
            ξ, w = EstimationToolkit.gauss_hermite_points(n)
            @test sum(w) ≈ 1
            # E[Z^k] for even k is (k-1)!!, zero for odd k; exact up to degree 2n - 1
            for k in 0:(2n - 1)
                exact = isodd(k) ? 0.0 : prod(1:2:max(k - 1, 1))
                scale = max(1, sum(w .* abs.(ξ) .^ k))  # odd moments cancel large terms
                @test sum(w .* ξ .^ k) ≈ exact atol = 1e-12 * scale
            end
        end
        @test_throws ArgumentError QuadratureKalmanFilter(@SVector(zeros(2)), SMatrix{2,2}(1.0I); quadrature_order=0)
    end

    # ------------------------------------------------------------------
    # 2. Second-order EKF: moments of quadratic functions, against Monte Carlo
    # ------------------------------------------------------------------
    @testset "Second-order EKF: quadratic moments" begin
        rng = MersenneTwister(5)
        x̄ = @SVector [1.0, -0.5]
        P = @SMatrix [0.30 0.10; 0.10 0.20]
        L = cholesky(Matrix(P)).L
        X = [x̄ + SVector{2}(L * randn(rng, 2)) for _ in 1:400_000]

        # measurement: predicted mean and innovation covariance
        meas = NSQuadMeas(0.1)
        hs = [measure(meas, x)[1] for x in X]
        f = SecondOrderExtendedKalmanFilter(x̄, P)
        y = SVector(2.0)
        update!(f, meas, y)
        ŷ = y[1] - innovation(f)[1]
        S = innovation_covariance(f)[1, 1]
        @test ŷ ≈ mean(hs) rtol = 5e-3
        @test S ≈ var(hs) + 0.1^2 rtol = 1e-2

        # dynamics: predicted mean and covariance
        f = SecondOrderExtendedKalmanFilter(x̄, P)
        predict!(f, NSQuadDyn(), nothing, 1.0)
        Y = [EstimationToolkit.propagate(NSQuadDyn(), x, nothing, 1.0) for x in X]
        μY = mean(Y)
        CY = sum((y - μY) * (y - μY)' for y in Y) / (length(Y) - 1)
        @test state_estimate(f) ≈ μY rtol = 5e-3
        @test Matrix(covariance(f)) ≈ Matrix(CY) rtol = 2e-2
    end

    # ------------------------------------------------------------------
    # 3. Singular process noise
    # ------------------------------------------------------------------
    @testset "Singular process noise" begin
        dyn = NSAccelNoise(1e-3)
        Q = Matrix(process_noise_cov(dyn, 10.0))
        @test rank(Q) == 3

        L = psd_sqrt(Q)
        @test L * L' ≈ Q rtol = 1e-10 atol = 1e-18
        Ap = randn(MersenneTwister(1), 4, 4)
        @test psd_sqrt(Ap * Ap' + I) * psd_sqrt(Ap * Ap' + I)' ≈ Ap * Ap' + I

        rng = MersenneTwister(3)
        V = [sample_process_noise(dyn, 10.0, rng) for _ in 1:50_000]
        @test all(v -> v isa SVector{6}, V)
        Cv = sum(v * v' for v in V) / length(V)
        @test Cv ≈ Q rtol = 3e-2
        Nq = nullspace(Q)
        # nothing outside the range of Q, up to rounding
        @test maximum(abs, reduce(hcat, [Nq' * v for v in V])) < 1e-6 * maximum(norm, V)

        # particle filters run with it (they used to throw at the first prediction)
        meas = NSPosXY(0.5)
        x0 = @SVector zeros(6)
        P0 = SMatrix{6,6}(Diagonal([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]))
        for pf in (BootstrapParticleFilter(Vector(x0), Matrix(P0), 500),
                   RegularizedParticleFilter(x0, P0, 500))
            for _ in 1:3
                predict!(pf, dyn, nothing, 10.0)
                update!(pf, meas, @SVector [0.1, -0.1])
            end
            @test all(isfinite, state_estimate(pf))
        end
    end
end
