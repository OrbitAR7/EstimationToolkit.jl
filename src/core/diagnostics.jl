"""
    nis(filter::AbstractFilter)
    normalized_innovation_squared(filter::AbstractFilter)

Compute the Normalized Innovation Squared (NIS) statistic.

For a consistent filter, NIS should follow a χ² distribution with n_y degrees of freedom,
where n_y is the measurement dimension.

Returns: ν' S⁻¹ ν where ν is the innovation and S is its covariance.
"""
function nis(filter::AbstractFilter)
    ν = innovation(filter)
    S = innovation_covariance(filter)
    return dot(ν, S \ ν)
end

const normalized_innovation_squared = nis

"""
    nees(x_true, x_est, P)
    normalized_estimation_error_squared(x_true, x_est, P)

Compute the Normalized Estimation Error Squared (NEES) statistic.

For a consistent filter, NEES should follow a χ² distribution with n_x degrees of freedom,
where n_x is the state dimension.

# Arguments
- `x_true`: True state
- `x_est`: Estimated state
- `P`: Estimation error covariance

Returns: e' P⁻¹ e where e = x_true - x_est
"""
function nees(x_true, x_est, P)
    e = x_true - x_est
    return dot(e, P \ e)
end

const normalized_estimation_error_squared = nees

"""
    consistency_test(statistic_values, dof; α=0.05)

Perform a χ² consistency test on a sequence of NIS or NEES values.

# Arguments
- `statistic_values`: Vector of NIS or NEES values
- `dof`: Degrees of freedom (n_y for NIS, n_x for NEES)
- `α`: Significance level (default 0.05 for 95% confidence)

Returns a named tuple with:
- `passed`: Boolean indicating if the test passed
- `mean_stat`: Mean of the statistics
- `lower_bound`: Lower confidence bound
- `upper_bound`: Upper confidence bound
- `fraction_in_bounds`: Fraction of individual values within bounds
"""
function consistency_test(statistic_values::AbstractVector, dof::Int; α=0.05)
    dist = Chisq(dof)
    lower = quantile(dist, α/2)
    upper = quantile(dist, 1 - α/2)
    
    mean_stat = mean(statistic_values)
    
    # Check how many individual values fall within bounds
    in_bounds = count(lower ≤ x ≤ upper for x in statistic_values)
    frac_in_bounds = in_bounds / length(statistic_values)
    
    # The mean should be close to dof (expected value of χ²)
    # We use a somewhat loose criterion here - within the confidence interval
    passed = (lower ≤ mean_stat ≤ upper) && (frac_in_bounds > 1 - α)
    
    return (
        passed = passed,
        mean_stat = mean_stat,
        expected = dof,
        lower_bound = lower,
        upper_bound = upper,
        fraction_in_bounds = frac_in_bounds
    )
end

"""
    chisq_mean_bounds(dof, n; α=0.05)

Two-sided bounds for the average of `n` independent χ² variables with `dof`
degrees of freedom each: the sum is χ² with `n·dof` degrees of freedom, so the
average lies in

    [quantile(Chisq(n·dof), α/2), quantile(Chisq(n·dof), 1 - α/2)] / n

with probability 1 - α. This is the acceptance region for the NEES (dof = n_x) or
NIS (dof = n_y) averaged over `n` Monte Carlo runs (Bar-Shalom, Li & Kirubarajan
2001, §5.4). Returns `(lower, upper)`; `(NaN, NaN)` for `n = 0`.
"""
function chisq_mean_bounds(dof::Integer, n::Integer; α=0.05)
    dof ≥ 1 || throw(ArgumentError("dof must be at least 1 (got $dof)"))
    n ≥ 0 || throw(ArgumentError("n must be non-negative (got $n)"))
    0 < α < 1 || throw(ArgumentError("α must be in (0, 1) (got $α)"))
    n == 0 && return (NaN, NaN)
    dist = Chisq(n * dof)
    return (quantile(dist, α / 2) / n, quantile(dist, 1 - α / 2) / n)
end

"""
    monte_carlo_consistency(runs, dof; α=0.05)
    monte_carlo_consistency(runs, bins, dof; α=0.05, nbins=nothing, within_run=:mean)

Run-averaged NEES or NIS from independent Monte Carlo runs, group by group, with
the χ² bounds the average should lie in if the filter is consistent.
[`consistency_test`](@ref) gives one verdict for one sequence; this shows *where*
along the trajectory a filter is over- or under-confident.

# Arguments
- `runs`: `runs[i][k]` is the NEES (or NIS) of run `i` at sample `k`. Runs may have
  different lengths. `missing` and non-finite values are skipped, so a step without
  a measurement can be stored as `NaN`. A matrix is also accepted, with one row per
  run.
- `bins`: `bins[i][k]` is the integer group index (1, 2, …) of that sample, for
  grouping by something other than the time step: an altitude band, a range gate,
  time since the last measurement. Same shape as `runs`. Samples with an index
  below 1 are left out. Without `bins`, sample `k` of every run goes to group `k`.
- `dof`: degrees of freedom of one sample (n_x for NEES, n_y for NIS).

# Keywords
- `α`: significance level; the bounds hold with probability 1 - α.
- `nbins`: number of groups. Defaults to the largest index in `bins`; an index
  above `nbins` is an error.
- `within_run`: what a run contributes to a group in which it has several samples.
  Runs are independent, but the samples of one run usually are not (NEES is
  correlated in time), and the bounds assume independent values.
  - `:mean` (default): the mean of its samples there. Each run counts once. The
    bounds are then conservative (too wide) by an amount that depends on how
    correlated the samples are, and exact when they are fully correlated.
  - `:first`: its first sample there. Each run counts once and the bounds are exact.
  - `:all`: every sample counts as an independent value. Right for the NIS of a
    consistent filter (white innovations); too narrow for NEES.

  When grouping by time step a run has one sample per group, so the three agree.

# Returns
A named tuple of vectors, one entry per group:
- `mean`: average over the values in the group (`NaN` for an empty group)
- `lower`, `upper`: bounds from [`chisq_mean_bounds`](@ref) for `count` values
- `count`: number of values averaged (runs, or samples with `within_run = :all`)
- `std_error`: standard error of the mean from the scatter of the values (`NaN` if
  fewer than two), which does not rely on the χ² assumption
- `inside`: whether `lower ≤ mean ≤ upper`

and the scalars `expected` (= `dof`), `α`, and `fraction_inside` (share of
non-empty groups inside the bounds; about 1 - α for a consistent filter).

# Example
```julia
# 100 runs, 4 states; nees[i][k] from nees(x_true, state_estimate(f), covariance(f))
c = monte_carlo_consistency(nees, 4)
overconfident = findall(c.mean .> c.upper)       # time steps where P is too small

# The same runs grouped into 5 km altitude bands
band = [[clamp(Int(fld(h, 5e3)) + 1, 1, 12) for h in alt] for alt in altitude]
c = monte_carlo_consistency(nees, band, 4)
```
"""
function monte_carlo_consistency(runs::AbstractVector, bins::AbstractVector, dof::Integer;
                                 α=0.05, nbins::Union{Nothing, Integer}=nothing,
                                 within_run::Symbol=:mean)
    dof ≥ 1 || throw(ArgumentError("dof must be at least 1 (got $dof)"))
    0 < α < 1 || throw(ArgumentError("α must be in (0, 1) (got $α)"))
    within_run in (:mean, :first, :all) ||
        throw(ArgumentError("within_run must be :mean, :first or :all (got :$within_run)"))
    length(runs) == length(bins) ||
        throw(ArgumentError("runs and bins must have the same number of runs " *
                            "(got $(length(runs)) and $(length(bins)))"))

    G = nbins === nothing ? _largest_bin(bins) : Int(nbins)
    G ≥ 0 || throw(ArgumentError("nbins must be non-negative (got $G)"))

    total = zeros(G)        # sum of the values in each group
    total_sq = zeros(G)     # sum of their squares, for the standard error
    n_values = zeros(Int, G)
    run_sum = zeros(G)      # this run's samples in each group
    run_n = zeros(Int, G)

    for (i, (r, b)) in enumerate(zip(runs, bins))
        length(r) == length(b) ||
            throw(ArgumentError("run $i has $(length(r)) values but $(length(b)) bin indices"))
        fill!(run_sum, 0.0)
        fill!(run_n, 0)
        for (v, g) in zip(r, b)
            (ismissing(v) || ismissing(g)) && continue
            isfinite(v) || continue
            g < 1 && continue
            g ≤ G || throw(ArgumentError("bin index $g in run $i exceeds nbins = $G"))
            if within_run === :all
                total[g] += v
                total_sq[g] += v^2
                n_values[g] += 1
            elseif within_run === :mean || run_n[g] == 0
                run_sum[g] += v
                run_n[g] += 1
            end
        end
        within_run === :all && continue
        for g in 1:G
            run_n[g] == 0 && continue
            v = run_sum[g] / run_n[g]
            total[g] += v
            total_sq[g] += v^2
            n_values[g] += 1
        end
    end

    avg = [n_values[g] > 0 ? total[g] / n_values[g] : NaN for g in 1:G]
    bounds = [chisq_mean_bounds(dof, n_values[g]; α=α) for g in 1:G]
    lower = first.(bounds)
    upper = last.(bounds)
    # Sample variance of the values (n - 1 in the denominator) over n.
    se = map(1:G) do g
        n = n_values[g]
        n ≥ 2 || return NaN
        return sqrt(max(total_sq[g] - n * avg[g]^2, 0.0) / (n - 1) / n)
    end
    inside = [n_values[g] > 0 && lower[g] ≤ avg[g] ≤ upper[g] for g in 1:G]
    n_groups = count(>(0), n_values)

    return (
        mean = avg,
        lower = lower,
        upper = upper,
        count = n_values,
        std_error = se,
        inside = inside,
        expected = dof,
        α = α,
        fraction_inside = n_groups > 0 ? count(inside) / n_groups : NaN
    )
end

# Group by time step: sample k of every run goes to group k.
function monte_carlo_consistency(runs::AbstractVector, dof::Integer; α=0.05)
    steps = [1:length(r) for r in runs]
    return monte_carlo_consistency(runs, steps, dof; α=α)
end

# One row per run; pad shorter runs with NaN or missing.
monte_carlo_consistency(values::AbstractMatrix, dof::Integer; kwargs...) =
    monte_carlo_consistency(collect(eachrow(values)), dof; kwargs...)

monte_carlo_consistency(values::AbstractMatrix, bins::AbstractMatrix, dof::Integer; kwargs...) =
    monte_carlo_consistency(collect(eachrow(values)), collect(eachrow(bins)), dof; kwargs...)

function _largest_bin(bins)
    G = 0
    for b in bins, g in b
        ismissing(g) && continue
        G = max(G, Int(g))
    end
    return G
end

"""
    mahalanobis_distance(x, μ, Σ)

Compute the Mahalanobis distance between point x and distribution N(μ, Σ).

Returns: √((x - μ)' Σ⁻¹ (x - μ))
"""
function mahalanobis_distance(x, μ, Σ)
    diff = x - μ
    return sqrt(dot(diff, Σ \ diff))
end
