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
    mahalanobis_distance(x, μ, Σ)

Compute the Mahalanobis distance between point x and distribution N(μ, Σ).

Returns: √((x - μ)' Σ⁻¹ (x - μ))
"""
function mahalanobis_distance(x, μ, Σ)
    diff = x - μ
    return sqrt(dot(diff, Σ \ diff))
end
