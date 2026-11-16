using Test
using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random
using Statistics

@testset "Adaptive Gaussian mixture" begin

    # ------------------------------------------------------------------
    # Models used below
    # ------------------------------------------------------------------

    # Linear: constant velocity, position measured
    struct AMLinear <: AbstractDynamicsModel{2, 0, 2}
        q::Float64
    end
    EstimationToolkit.propagate(::AMLinear, x, u, dt) = SVector(x[1] + dt * x[2], x[2])
    EstimationToolkit.process_noise_cov(m::AMLinear, dt) =
        m.q * @SMatrix [dt^3/3 dt^2/2; dt^2/2 dt]

    struct AMPosition <: AbstractMeasurementModel{2, 1, 1}
        σ::Float64
    end
    EstimationToolkit.measure(::AMPosition, x) = SVector(x[1])
    EstimationToolkit.measurement_noise_cov(m::AMPosition) = SMatrix{1,1}(m.σ^2)

    # Nonlinear: state (range, bearing) mapped to Cartesian. A Gaussian that is
    # wide in bearing becomes a crescent.
    struct AMPolarToCartesian <: AbstractDynamicsModel{2, 0, 2} end
    EstimationToolkit.propagate(::AMPolarToCartesian, x, u, dt) =
        SVector(x[1] * cos(x[2]), x[1] * sin(x[2]))
    EstimationToolkit.process_noise_cov(::AMPolarToCartesian, dt) = @SMatrix zeros(2, 2)

    # Identity dynamics, used to exercise updates on a prepared mixture
    struct AMIdentity <: AbstractDynamicsModel{2, 0, 2} end
    EstimationToolkit.propagate(::AMIdentity, x, u, dt) = x
    EstimationToolkit.process_noise_cov(::AMIdentity, dt) = @SMatrix zeros(2, 2)

    # moments of a weighted set of Gaussians
    mix_mean(w, μ) = sum(w[i] * μ[i] for i in eachindex(w))
    function mix_cov(w, μ, P)
        m = mix_mean(w, μ)
        return sum(w[i] * (P[i] + (μ[i] - m) * (μ[i] - m)') for i in eachindex(w))
    end

    # ------------------------------------------------------------------
    @testset "split_gaussian" begin
        μ = SVector(1.0, -2.0, 0.5)
        A = @SMatrix [2.0 0.3 -0.1; 0.3 1.0 0.4; -0.1 0.4 0.5]
        Σ = A * A'

        @testset "moments are preserved (shrink = $shrink)" for shrink in (0.3, 0.5, 0.7, 0.9)
            for S in (cholesky(Symmetric(Σ)).L, principal_sqrt(Σ)), k in 1:3
                s = SVector{3}(S[:, k])
                @test s' * (Σ \ s) ≈ 1.0                      # a one-sigma direction
                means, Σs, w = split_gaussian(μ, Σ, s; shrink = shrink)
                @test sum(w) ≈ 1.0
                @test mix_mean(w, means) ≈ μ
                @test mix_cov(w, means, (Σs, Σs, Σs)) ≈ Σ
                @test is_positive_definite(Σs)
                # variance along s is reduced by shrink², untouched across it
                @test s' * (Σ \ (Σs * (Σ \ s))) ≈ shrink^2
            end
        end

        @testset "arbitrary direction, normalized" begin
            v = SVector(0.2, -1.0, 3.0)
            s = v / sqrt(v' * (Σ \ v))
            means, Σs, w = split_gaussian(μ, Σ, s)
            @test mix_cov(w, means, (Σs, Σs, Σs)) ≈ Σ
        end

        @testset "1-D: matches the standard normal through the fifth moment" begin
            for shrink in (0.4, 0.7, 0.85)
                means, Σs, w = split_gaussian(SVector(0.0), SMatrix{1,1}(1.0), SVector(1.0);
                                              shrink = shrink)
                m = [x[1] for x in means]
                v = Σs[1, 1]
                @test v ≈ shrink^2
                # raw moments of N(m, v): E x² = m² + v, E x⁴ = m⁴ + 6m²v + 3v²
                @test sum(w[i] * (m[i]^2 + v) for i in 1:3) ≈ 1.0
                @test sum(w[i] * (m[i]^3 + 3m[i] * v) for i in 1:3) ≈ 0.0 atol = 1e-14
                @test sum(w[i] * (m[i]^4 + 6m[i]^2 * v + 3v^2) for i in 1:3) ≈ 3.0
                @test sum(w[i] * (m[i]^5 + 10m[i]^3 * v + 15m[i] * v^2) for i in 1:3) ≈ 0.0 atol = 1e-13
            end
        end

        @testset "1-D: density stays close to the original at the default shrink" begin
            means, Σs, w = split_gaussian(SVector(0.0), SMatrix{1,1}(1.0), SVector(1.0))
            σ = sqrt(Σs[1, 1])
            mix(x) = sum(w[i] * exp(-0.5 * ((x - means[i][1]) / σ)^2) / (σ * sqrt(2π)) for i in 1:3)
            ref(x) = exp(-0.5 * x^2) / sqrt(2π)
            xs = range(-6, 6; length = 2401)
            dx = step(xs)
            @test sum(mix.(xs)) * dx ≈ 1.0 atol = 1e-6
            @test maximum(abs.(mix.(xs) .- ref.(xs))) < 0.06 * ref(0.0)   # within 6 % of the peak
            l2 = sqrt(sum((mix.(xs) .- ref.(xs)) .^ 2) * dx)
            @test l2 < 0.05 * sqrt(sum(ref.(xs) .^ 2) * dx)
        end

        @testset "GaussianBelief method" begin
            s = SVector{3}(principal_sqrt(Σ)[:, 3])
            comps, w = split_gaussian(GaussianBelief(μ, Σ), s)
            @test length(comps) == 3
            @test mix_mean(w, [c.μ for c in comps]) ≈ μ
        end
    end

    # ------------------------------------------------------------------
    @testset "Two-way split_component keeps the covariance" begin
        μ = SVector(1.0, 2.0)
        Σ = @SMatrix [4.0 1.2; 1.2 1.0]
        for method in (:eigenvalue, :axis), f in (0.3, 0.5, 0.9)
            c1, c2, w1, w2 = split_component(GaussianBelief(μ, Σ), 0.4; method = method, split_factor = f)
            @test w1 + w2 ≈ 0.4
            w = [w1, w2] ./ 0.4
            @test mix_mean(w, [c1.μ, c2.μ]) ≈ μ
            @test mix_cov(w, [c1.μ, c2.μ], [c1.Σ, c2.Σ]) ≈ Σ
            @test is_positive_definite(c1.Σ)
        end
        @test_throws AssertionError split_component(GaussianBelief(μ, Σ), 1.0; split_factor = 1.0)

        # should_split used to throw (its local variable shadowed `eigvals`)
        @test EstimationToolkit.should_split(GaussianBelief(μ, Σ), 1.0) == true
        @test EstimationToolkit.should_split(GaussianBelief(μ, Σ), 10.0) == false
    end

    # ------------------------------------------------------------------
    @testset "principal_sqrt" begin
        P = @SMatrix [1.0e4 3.0 0.02; 3.0 1.0e-2 5.0e-5; 0.02 5.0e-5 1.0e-6]   # correlations 0.3, 0.2, 0.5
        @test is_positive_definite(P)
        S = principal_sqrt(P)
        @test S * S' ≈ P
        # unit independence: rescaling the state rescales the square root
        D = SMatrix{3,3}(Diagonal(SVector(1e-3, 1.0, 1e3)))
        S2 = principal_sqrt(D * P * D)
        # columns are defined up to sign
        for k in 1:3
            @test (S2[:, k] ≈ D * S[:, k]) || (S2[:, k] ≈ -(D * S[:, k]))
        end
    end

    # ------------------------------------------------------------------
    @testset "Linear system: equals the Kalman filter and never splits" begin
        dyn = AMLinear(0.05)
        meas = AMPosition(0.4)
        x0 = SVector(0.5, -0.2)
        P0 = @SMatrix [2.0 0.3; 0.3 1.0]

        kf = KalmanFilter(x0, P0)
        f = AdaptiveGaussianMixtureFilter(x0, P0; split_tol = 1e-6)
        rng = MersenneTwister(11)
        x = SVector(0.0, 1.0)
        for k in 1:200
            x = EstimationToolkit.propagate(dyn, x, nothing, 0.5)
            y = SVector(x[1] + 0.4 * randn(rng))
            predict!(kf, dyn, nothing, 0.5); update!(kf, meas, y)
            predict!(f, dyn, nothing, 0.5); update!(f, meas, y)
        end
        @test num_components(f) == 1
        @test f.n_splits == 0
        @test state_estimate(f) ≈ state_estimate(kf) rtol = 1e-9
        @test covariance(f) ≈ covariance(kf) rtol = 1e-9
        @test innovation(f) ≈ innovation(kf) rtol = 1e-8
        @test innovation_covariance(f) ≈ innovation_covariance(kf) rtol = 1e-8

        # propagation only, with a tolerance far below any real nonlinearity
        g = AdaptiveGaussianMixtureFilter(x0, P0; split_tol = 1e-9)
        for k in 1:500
            predict!(g, dyn, nothing, 0.5)
        end
        @test num_components(g) == 1
        @test maximum(abs, g.entropy_dev) < 1e-10
    end

    # ------------------------------------------------------------------
    @testset "Entropy difference detects nonlinearity" begin
        dyn = AMPolarToCartesian()
        x0 = SVector(10.0, 0.0)
        rule = EstimationToolkit._sigma_rule(AdaptiveGaussianMixtureFilter(x0, SMatrix{2,2}(1.0I)))
        g = x -> EstimationToolkit.propagate(dyn, x, nothing, 1.0)

        devs = map((0.01, 0.1, 0.3, 0.6)) do σθ
            P = SMatrix{2,2}(Diagonal(SVector(0.1^2, σθ^2)))
            EstimationToolkit._entropy_difference(EstimationToolkit._sigma_transform(g, x0, P, rule))
        end
        @test all(d -> d >= 0, devs)            # cubature rule: P_σ - P_lin is PSD
        @test issorted(devs)                    # wider in bearing, more nonlinear
        @test devs[1] < 1e-4
        @test devs[end] > 0.05
        # For a small bearing spread σθ the curvature moves the mean radially by
        # about r σθ²/2, and the entropy difference compares the square of that with
        # the (fixed) range variance: fourth order in σθ. Ten times the spread gives
        # about 10⁴ times the difference.
        @test 0.8e4 < devs[2] / devs[1] < 1.2e4

        # the nonlinearity is in the bearing direction, and that is the one picked
        P = SMatrix{2,2}(Diagonal(SVector(0.1^2, 0.3^2)))
        r = EstimationToolkit._sigma_transform(g, x0, P, rule)
        s = r.S[:, EstimationToolkit._most_nonlinear_direction(r)]
        @test abs(s[2]) > 100 * abs(s[1])
    end

    # ------------------------------------------------------------------
    @testset "Crescent: the split mixture describes the true distribution" begin
        dyn = AMPolarToCartesian()
        x0 = SVector(10.0, 0.0)
        P0 = SMatrix{2,2}(Diagonal(SVector(0.2^2, 0.5^2)))     # ±0.5 rad in bearing

        # truth by Monte Carlo
        rng = MersenneTwister(2026)
        L = cholesky(Symmetric(P0)).L
        samples = [EstimationToolkit.propagate(dyn, x0 + L * SVector(randn(rng), randn(rng)), nothing, 1.0)
                   for _ in 1:20_000]
        mc_mean = mean(samples)
        mc_cov = cov(reduce(hcat, samples)')

        single = AdaptiveGaussianMixtureFilter(x0, P0; split_tol = Inf)
        predict!(single, dyn, nothing, 1.0)
        @test num_components(single) == 1

        f = AdaptiveGaussianMixtureFilter(x0, P0; split_tol = 0.01, max_components = 60)
        predict!(f, dyn, nothing, 1.0)
        @test num_components(f) > 3
        @test num_components(f) == 1 + 2 * f.n_splits
        @test sum(f.weights) ≈ 1.0
        # 60 components are not enough to bring this case within 0.01 nats, but the
        # worst component is far better off than the single Gaussian was
        @test maximum(f.entropy_dev) < 0.2 * single.entropy_dev[1]

        # an easier case is refined until every component is within tolerance
        easy = AdaptiveGaussianMixtureFilter(x0, SMatrix{2,2}(Diagonal(SVector(0.5^2, 0.2^2)));
                                             split_tol = 0.02, max_components = 60)
        predict!(easy, dyn, nothing, 1.0)
        @test 1 < num_components(easy) < 60
        @test maximum(easy.entropy_dev) <= 0.02

        # average log-density of the true samples: higher is better
        score(filter) = mean(mixture_logpdf(filter, x) for x in samples)
        @test score(f) > score(single) + 0.2

        # the mixture's mean and covariance are at least as close to Monte Carlo
        @test norm(state_estimate(f) - mc_mean) < 0.05
        @test norm(covariance(f) - mc_cov) / norm(mc_cov) < 0.05
        @test norm(covariance(f) - mc_cov) <= norm(covariance(single) - mc_cov)

        # density integrates to one over the plane
        xs = range(-2, 16; length = 361); ys = range(-12, 12; length = 481)
        mass = sum(exp(mixture_logpdf(f, SVector(x, y))) for x in xs, y in ys) * step(xs) * step(ys)
        @test mass ≈ 1.0 atol = 1e-3
    end

    # ------------------------------------------------------------------
    @testset "predict! splits on its own when the tolerance is exceeded" begin
        dyn = AMPolarToCartesian()
        x0 = SVector(10.0, 0.0)
        P0 = SMatrix{2,2}(Diagonal(SVector(0.2^2, 0.5^2)))

        # room for exactly one split
        f = AdaptiveGaussianMixtureFilter(x0, P0; split_tol = 0.01, max_components = 3)
        predict!(f, dyn, nothing, 1.0)
        @test num_components(f) == 3
        @test f.n_splits == 1
        @test sum(f.weights) ≈ 1.0
        @test sort(f.weights) ≈ [1/6, 1/6, 2/3]

        # splitting leaves the prior's moments alone
        prior = AdaptiveGaussianMixtureFilter(x0, P0)
        split!(prior, 1, SVector{2}(principal_sqrt(P0)[:, 1]))
        split!(prior, 2, SVector{2}(principal_sqrt(prior.covs[2])[:, 2]))
        @test num_components(prior) == 5
        @test state_estimate(prior) ≈ x0
        @test covariance(prior) ≈ P0

        # at the cap nothing is split
        capped = AdaptiveGaussianMixtureFilter(x0, P0; split_tol = 0.01, max_components = 2)
        predict!(capped, dyn, nothing, 1.0)
        @test num_components(capped) == 1
        @test capped.entropy_dev[1] > 0.01      # the deviation is still recorded

        # and with splitting off it is one sigma-point filter
        off = AdaptiveGaussianMixtureFilter(x0, P0; split_tol = Inf)
        predict!(off, dyn, nothing, 1.0)
        @test num_components(off) == 1
    end

    # ------------------------------------------------------------------
    @testset "Update: weights follow Bayes' rule" begin
        meas = AMPosition(0.5)
        μ1, μ2 = SVector(-1.0, 0.0), SVector(2.0, 0.0)
        P = SMatrix{2,2}(1.0I)
        f = AdaptiveGaussianMixtureFilter([μ1, μ2], [P, P], [0.3, 0.7]; merge_tol = 0.0)
        y = 0.2
        update!(f, meas, SVector(y))
        S = 1.0 + 0.25
        L = [0.3 * exp(-0.5 * (y - μ1[1])^2 / S), 0.7 * exp(-0.5 * (y - μ2[1])^2 / S)]
        @test f.weights ≈ L ./ sum(L)
        # each component is a Kalman update of its own prior
        K = 1.0 / S
        @test f.means[1][1] ≈ μ1[1] + K * (y - μ1[1])
        @test f.covs[1][1, 1] ≈ 1.0 - K
        @test f.covs[1][2, 2] ≈ 1.0
        # mixture innovation: predicted measurement is a two-component mixture
        ŷ = 0.3 * μ1[1] + 0.7 * μ2[1]
        @test innovation(f)[1] ≈ y - ŷ
        @test innovation_covariance(f)[1, 1] ≈ S + 0.3 * (μ1[1] - ŷ)^2 + 0.7 * (μ2[1] - ŷ)^2

        # tight measurement, both components far away: no underflow
        tight = AMPosition(1e-3)
        Pt = SMatrix{2,2}(Diagonal(SVector(1e-6, 1.0)))
        g = AdaptiveGaussianMixtureFilter([SVector(0.0, 0.0), SVector(10.0, 0.0)], [Pt, Pt], [0.5, 0.5])
        update!(g, tight, SVector(4.0))
        @test num_components(g) == 1
        @test abs(state_estimate(g)[1] - 2.0) < 1e-6
    end

    # ------------------------------------------------------------------
    @testset "Reduction" begin
        P = SMatrix{2,2}(1.0I)

        @testset "merge_cost" begin
            μ = SVector(0.3, -1.0)
            @test merge_cost(0.2, μ, P, 0.5, μ, P) ≈ 0.0 atol = 1e-12     # identical
            near = merge_cost(0.5, μ, P, 0.5, μ + SVector(0.1, 0.0), P)
            far = merge_cost(0.5, μ, P, 0.5, μ + SVector(3.0, 0.0), P)
            @test 0 < near < far
            @test merge_cost(0.5, μ, P, 0.5, μ + SVector(3.0, 0.0), P) ≈
                  merge_cost(0.5, μ + SVector(3.0, 0.0), P, 0.5, μ, P)     # symmetric
            # closed form for equal covariances in 1-D: ½ w log(1 + d²/4)
            c = merge_cost(0.5, SVector(0.0), SMatrix{1,1}(1.0), 0.5, SVector(2.0), SMatrix{1,1}(1.0))
            @test c ≈ 0.5 * log(1 + 2.0^2 / 4)
        end

        @testset "merging keeps the mixture moments" begin
            means = [SVector(0.0, 0.0), SVector(0.05, 0.0), SVector(4.0, 1.0), SVector(4.0, 1.02), SVector(-3.0, 2.0)]
            covs = [P, P, 0.5 * P, 0.5 * P, 2.0 * P]
            w = [0.2, 0.2, 0.25, 0.15, 0.2]
            f = AdaptiveGaussianMixtureFilter(means, covs, w; merge_tol = 1e-3)
            m0, C0 = state_estimate(f), covariance(f)
            reduce_components!(f)
            @test num_components(f) == 3          # the two near-duplicates merged
            @test f.n_merges == 2
            @test sum(f.weights) ≈ 1.0
            @test state_estimate(f) ≈ m0
            @test covariance(f) ≈ C0

            # forced reduction below max_components
            g = AdaptiveGaussianMixtureFilter(means, covs, w; merge_tol = 0.0, max_components = 5)
            g.max_components = 2
            reduce_components!(g)
            @test num_components(g) == 2
            @test state_estimate(g) ≈ m0
            @test covariance(g) ≈ C0
        end

        @testset "pruning" begin
            f = AdaptiveGaussianMixtureFilter([SVector(0.0, 0.0), SVector(5.0, 0.0)], [P, P],
                                              [1 - 1e-10, 1e-10]; weight_threshold = 1e-8)
            prune_components!(f)
            @test num_components(f) == 1
            @test f.weights == [1.0]
        end

        @testset "components that agree after an update are merged" begin
            # A split prior followed by a precise measurement of the full state
            # direction leaves three near-identical posteriors.
            f = AdaptiveGaussianMixtureFilter(SVector(0.0, 0.0), SMatrix{2,2}(1.0I))
            split!(f, 1, SVector(1.0, 0.0))
            @test num_components(f) == 3
            @test state_estimate(f) ≈ SVector(0.0, 0.0) atol = 1e-14
            @test covariance(f) ≈ SMatrix{2,2}(1.0I)
            update!(f, AMPosition(1e-4), SVector(0.3))
            @test num_components(f) == 1
            @test state_estimate(f)[1] ≈ 0.3 atol = 1e-6
        end
    end

    # ------------------------------------------------------------------
    @testset "Mixture density and conversions" begin
        P = SMatrix{2,2}(1.0I)
        f = AdaptiveGaussianMixtureFilter(SVector(1.0, 2.0), P)
        x = SVector(0.5, 2.5)
        @test mixture_logpdf(f, x) ≈ -0.5 * (0.5^2 + 0.5^2) - log(2π)
        @test gaussian_logpdf(x, SVector(1.0, 2.0), P) ≈ mixture_logpdf(f, x)
        # far in the tail the density underflows but its log does not
        far = SVector(1.0e3, 0.0)
        @test isfinite(mixture_logpdf(f, far))
        @test mixture_logpdf(f, far) < -1.0e5

        # batch method agrees with the single-point one on a real mixture
        g = AdaptiveGaussianMixtureFilter([SVector(0.0, 0.0), SVector(3.0, 1.0)],
                                          [P, SMatrix{2,2}(2.0, 0.5, 0.5, 1.0)], [0.4, 0.6])
        pts = [SVector(0.1, 0.2), SVector(3.0, 1.0), SVector(-40.0, 25.0)]
        @test mixture_logpdf(g, pts) ≈ [mixture_logpdf(g, x) for x in pts]

        gmm = GMMBelief(f)
        @test length(gmm.components) == 1
        @test mixture_logpdf(gmm, x) ≈ mixture_logpdf(f, x)
        f2 = AdaptiveGaussianMixtureFilter(gmm)
        @test state_estimate(f2) ≈ state_estimate(f)

        # plain-array constructor
        f3 = AdaptiveGaussianMixtureFilter([1.0, 2.0], [1.0 0.0; 0.0 1.0])
        @test state_estimate(f3) ≈ SVector(1.0, 2.0)
        @test effective_components(f3) ≈ 1.0
    end
end
