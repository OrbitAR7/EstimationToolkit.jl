"""
Component splitting utilities for Gaussian mixture management.

Splitting is used when a component has high uncertainty (large covariance)
or when the filter needs to represent multi-modal distributions better.
"""

"""
    split_component(comp::GaussianBelief, weight::T; method=:eigenvalue) where T

Split a Gaussian component into two components along the direction of
maximum uncertainty.

# Arguments
- `comp`: Gaussian component to split
- `weight`: Weight of the original component

# Keyword Arguments
- `method`: Splitting method
  - `:eigenvalue` - Split along largest eigenvector (default)
  - `:axis` - Split along coordinate axes
- `split_factor`: Distance factor for splitting, in (0, 1) (default: 0.5)
  - New means are placed at μ ± split_factor * σ_max

The two components share the covariance Σ - δδᵀ, where δ is the mean offset, so
that the pair has the same mean and covariance as the original component. (A
split that keeps the full Σ on both halves inflates the mixture covariance by
δδᵀ.) For a split that also matches the higher moments, see [`split_gaussian`](@ref).

# Returns
- `(comp1, comp2, weight1, weight2)`: Two new components with equal weights

# Example
```julia
belief = GaussianBelief(x, P)
c1, c2, w1, w2 = split_component(belief, 0.5)
```
"""
function split_component(
    comp::GaussianBelief{T, N},
    weight::T;
    method::Symbol = :eigenvalue,
    split_factor::T = T(0.5)
) where {T<:Real, N}
    
    @assert 0 < split_factor < 1 "split_factor must be in (0, 1) so the split components keep a positive definite covariance"
    
    μ = comp.μ
    Σ = comp.Σ
    
    if method == :eigenvalue
        # Split along direction of maximum uncertainty
        eigvals, eigvecs = eigen(Symmetric(Σ))
        
        # Find largest eigenvalue
        max_idx = argmax(eigvals)
        λ_max = eigvals[max_idx]
        v_max = SVector{N, T}(eigvecs[:, max_idx])
        
        # Split distance
        σ = sqrt(abs(λ_max))
        δ = split_factor * σ * v_max
        
        # New means
        μ1 = μ + δ
        μ2 = μ - δ
        
        # Each component gets half the weight. Removing δδᵀ from the covariance
        # keeps the covariance of the pair equal to Σ.
        Σ_split = make_symmetric(Σ - δ * δ')
        comp1 = GaussianBelief(μ1, Σ_split)
        comp2 = GaussianBelief(μ2, Σ_split)
        w1 = w2 = weight / 2
        
    elseif method == :axis
        # Split along coordinate axis with maximum variance
        variances = [Σ[i,i] for i in 1:N]
        max_idx = argmax(variances)
        
        # Split distance
        σ = sqrt(variances[max_idx])
        # Offset along the conditional direction Σ[:, k] / σ_k rather than the bare
        # axis, so that Σ - δδᵀ stays positive definite when the state is correlated.
        δ = split_factor * Σ[:, max_idx] / σ
        
        # New means
        μ1 = μ + δ
        μ2 = μ - δ
        
        # Each component gets half the weight. Removing δδᵀ from the covariance
        # keeps the covariance of the pair equal to Σ.
        Σ_split = make_symmetric(Σ - δ * δ')
        comp1 = GaussianBelief(μ1, Σ_split)
        comp2 = GaussianBelief(μ2, Σ_split)
        w1 = w2 = weight / 2
        
    else
        error("Unknown splitting method: $method")
    end
    
    return comp1, comp2, w1, w2
end

"""
    split_component(ekf::ExtendedKalmanFilter, weight; kwargs...)

Split an EKF component for use in Gaussian Sum Filter.

Returns two new EKF instances and their weights.
"""
function split_component(
    ekf::ExtendedKalmanFilter{T, N},
    weight::T;
    kwargs...
) where {T<:Real, N}
    
    # Extract belief
    belief = GaussianBelief(state_estimate(ekf), covariance(ekf))
    
    # Split belief
    comp1, comp2, w1, w2 = split_component(belief, weight; kwargs...)
    
    # Create new EKFs
    ekf1 = ExtendedKalmanFilter(comp1.μ, comp1.Σ)
    ekf2 = ExtendedKalmanFilter(comp2.μ, comp2.Σ)
    
    return ekf1, ekf2, w1, w2
end

"""
    should_split(comp::GaussianBelief, threshold::Real)

Determine if a component should be split based on its uncertainty.

A component is split if its maximum eigenvalue exceeds the threshold.

# Arguments
- `comp`: Gaussian component to check
- `threshold`: Eigenvalue threshold for splitting

# Returns
- `true` if component should be split, `false` otherwise
"""
function should_split(comp::GaussianBelief{T, N}, threshold::Real) where {T<:Real, N}
    # (named λ, not eigvals: assigning to `eigvals` would shadow the function)
    λ = eigvals(Symmetric(Matrix(comp.Σ)))
    return maximum(λ) > threshold
end

"""
    split_high_uncertainty_components!(gsf::GaussianSumFilter, threshold)

Automatically split components with uncertainty above threshold.

This is useful for adaptive mixture refinement in highly nonlinear problems.

# Arguments
- `gsf`: Gaussian Sum Filter to modify
- `threshold`: Maximum eigenvalue threshold for splitting

# Example
```julia
# Split any component with σ_max > 10.0
split_high_uncertainty_components!(gsf, 100.0)
```
"""
function split_high_uncertainty_components!(
    gsf::GaussianSumFilter{T, N},
    threshold::T
) where {T<:Real, N}
    
    new_components = ExtendedKalmanFilter{T, N}[]
    new_weights = T[]
    
    for (comp, weight) in zip(gsf.components, gsf.weights)
        belief = GaussianBelief(state_estimate(comp), covariance(comp))
        
        if should_split(belief, threshold)
            # Split this component
            ekf1, ekf2, w1, w2 = split_component(comp, weight)
            push!(new_components, ekf1)
            push!(new_components, ekf2)
            push!(new_weights, w1)
            push!(new_weights, w2)
        else
            # Keep original
            push!(new_components, comp)
            push!(new_weights, weight)
        end
    end
    
    # Update GSF
    gsf.components = new_components
    gsf.weights = new_weights
    
    # Prune if too many components
    prune_components!(gsf)
    
    return gsf
end

"""
    split_gaussian(μ, Σ, s; shrink=0.7) -> (means, Σ_split, weights)

Replace the Gaussian N(μ, Σ) by three narrower Gaussians spread along the
one-sigma direction `s`, without changing the mean or the covariance of the whole.

`s` must be a one-sigma displacement of Σ, i.e. `s' * inv(Σ) * s == 1`. Any column
of a square root of Σ qualifies (a Cholesky column, or `sqrt(λ) * v` for an
eigenpair), and so does `v / sqrt(v' * inv(Σ) * v)` for an arbitrary direction `v`.

# Construction

In the coordinate ξ along `s` the original density is the standard normal. It is
replaced by

    1/6 N(-m, σ̃²) + 2/3 N(0, σ̃²) + 1/6 N(+m, σ̃²),    m = sqrt(3 (1 - σ̃²)),

with σ̃ = `shrink`. The weights 1/6, 2/3, 1/6 and the offsets ±m are the
three-point Gauss-Hermite rule scaled by the variance that is taken out of each
component, and with them the mixture matches the standard normal in every moment
up to the fifth for any σ̃ in (0, 1):

    variance:      σ̃² + 2 (1/6) m²           = σ̃² + (1 - σ̃²) = 1
    fourth moment: 3σ̃⁴ + 6σ̃²(1 - σ̃²) + 3(1 - σ̃²)² = 3

(odd moments vanish by symmetry). All three components get the covariance
`Σ - (1 - σ̃²) s s'`, which is Σ with its variance along `s` reduced by σ̃².

`shrink` trades resolution against fidelity: a small value gives narrow components
(better at following a curved density afterwards) but a mixture that is visibly
lumpy; a value near 1 barely changes anything. At the default 0.7 the mixture
density is within about 5 % of the original's peak value everywhere.

This is a moment-matching library. DeMars, Bishop & Jah (2013) instead tabulate
libraries that minimize the L2 distance to the standard normal; those fit the
density slightly better pointwise but do not preserve the variance exactly.

# Returns
- `means`: tuple of three `SVector`s, `(μ - m s, μ, μ + m s)`
- `Σ_split`: the covariance shared by the three components
- `weights`: `(1/6, 2/3, 1/6)`, to be multiplied by the weight of the component
  that was split
"""
function split_gaussian(μ::SVector{N, T}, Σ::SMatrix{N, N, T}, s::SVector{N, T};
                        shrink::Real = 0.7) where {N, T<:Real}
    @assert 0 < shrink < 1 "shrink must be in (0, 1)"
    c = T(1 - shrink^2)          # variance removed along s (in units of σ²)
    m = sqrt(3 * c)              # offset of the outer components (in units of σ)
    
    Σ_split = make_symmetric(Σ - c * (s * s'))
    means = (μ - m * s, μ, μ + m * s)
    weights = (T(1) / 6, T(2) / 3, T(1) / 6)
    
    return means, Σ_split, weights
end

"""
    split_gaussian(comp::GaussianBelief, s; shrink=0.7)

Same, returning three `GaussianBelief`s and the weights.
"""
function split_gaussian(comp::GaussianBelief{T, N}, s::SVector{N, T}; kwargs...) where {T<:Real, N}
    means, Σ_split, weights = split_gaussian(comp.μ, comp.Σ, s; kwargs...)
    return map(μ -> GaussianBelief(μ, Σ_split), means), weights
end
