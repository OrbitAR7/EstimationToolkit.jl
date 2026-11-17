"""
Run-averaged NEES/NIS across Monte Carlo runs: `chisq_mean_bounds` and
`monte_carlo_consistency`.
"""

using Test
using EstimationToolkit
using Distributions
using Random
using Statistics

@testset "Monte Carlo consistency" begin

    @testset "chisq_mean_bounds matches Distributions.Chisq" begin
        for (dof, n) in ((1, 1), (1, 200), (4, 50), (6, 100))
            lo, hi = chisq_mean_bounds(dof, n)
            @test lo ≈ quantile(Chisq(n * dof), 0.025) / n
            @test hi ≈ quantile(Chisq(n * dof), 0.975) / n
            @test lo < dof < hi
        end
        # A known value: 50 runs of a 1-dof statistic, α = 0.05
        lo, hi = chisq_mean_bounds(1, 50)
        @test lo ≈ 32.3574 / 50 atol = 1e-3
        @test hi ≈ 71.4202 / 50 atol = 1e-3
        # α changes the width
        lo99, hi99 = chisq_mean_bounds(1, 50; α = 0.01)
        @test lo99 < lo && hi99 > hi
        @test all(isnan, chisq_mean_bounds(3, 0))
        @test_throws ArgumentError chisq_mean_bounds(0, 10)
        @test_throws ArgumentError chisq_mean_bounds(2, -1)
        @test_throws ArgumentError chisq_mean_bounds(2, 10; α = 1.5)
    end

    @testset "Grouping by time step, unequal run lengths" begin
        runs = [[1.0, 2.0, 3.0], [3.0, 4.0], [5.0]]
        c = monte_carlo_consistency(runs, 1)
        @test c.count == [3, 2, 1]
        @test c.mean ≈ [3.0, 3.0, 3.0]
        @test c.expected == 1
        @test c.lower[1] ≈ chisq_mean_bounds(1, 3)[1]
        @test c.upper[2] ≈ chisq_mean_bounds(1, 2)[2]
        @test c.std_error[1] ≈ std([1.0, 3.0, 5.0]) / sqrt(3)
        @test isnan(c.std_error[3])
        # Matrix form, padded with NaN, gives the same answer
        M = [1.0 2.0 3.0; 3.0 4.0 NaN; 5.0 NaN NaN]
        cm = monte_carlo_consistency(M, 1)
        @test cm.count == c.count
        @test cm.mean ≈ c.mean
    end

    @testset "Missing and non-finite values are skipped" begin
        runs = [[1.0, NaN, 2.0], Union{Missing, Float64}[missing, 4.0, Inf]]
        c = monte_carlo_consistency(runs, 2)
        @test c.count == [1, 1, 1]
        @test c.mean ≈ [1.0, 4.0, 2.0]
    end

    @testset "Grouping by bins and the within_run options" begin
        runs = [[1.0, 3.0, 10.0], [2.0, 6.0, 20.0]]
        bins = [[1, 1, 2], [1, 1, 0]]          # bin 0 is left out
        cmean = monte_carlo_consistency(runs, bins, 1)
        @test cmean.count == [2, 1]
        @test cmean.mean ≈ [(2.0 + 4.0) / 2, 10.0]
        cfirst = monte_carlo_consistency(runs, bins, 1; within_run = :first)
        @test cfirst.mean ≈ [(1.0 + 2.0) / 2, 10.0]
        call = monte_carlo_consistency(runs, bins, 1; within_run = :all)
        @test call.count == [4, 1]
        @test call.mean ≈ [3.0, 10.0]
        # Empty group when nbins exceeds the largest index
        c3 = monte_carlo_consistency(runs, bins, 1; nbins = 3)
        @test c3.count[3] == 0 && isnan(c3.mean[3]) && !c3.inside[3]
        @test c3.fraction_inside == count(c3.inside[1:2]) / 2
        @test_throws ArgumentError monte_carlo_consistency(runs, bins, 1; nbins = 1)
        @test_throws ArgumentError monte_carlo_consistency(runs, bins, 1; within_run = :median)
        @test_throws ArgumentError monte_carlo_consistency(runs, [[1, 1]], 1)
        @test_throws ArgumentError monte_carlo_consistency([[1.0, 2.0]], [[1]], 1)
    end

    @testset "Consistent and overconfident statistics" begin
        rng = MersenneTwister(7)
        dof, N, K = 3, 400, 60
        consistent = [rand(rng, Chisq(dof), K) for _ in 1:N]
        c = monte_carlo_consistency(consistent, dof)
        @test all(c.count .== N)
        @test 0.85 ≤ c.fraction_inside ≤ 1.0
        @test abs(mean(c.mean) - dof) < 0.1
        # A covariance too small by a factor of 2 doubles NEES: every step is out.
        over = [2 .* r for r in consistent]
        co = monte_carlo_consistency(over, dof)
        @test co.fraction_inside == 0.0
        @test all(co.mean .> co.upper)
    end
end
