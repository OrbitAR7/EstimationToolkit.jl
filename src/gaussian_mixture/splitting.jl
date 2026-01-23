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
- `split_factor`: Distance factor for splitting (default: 1.0)
  - New means are placed at μ ± split_factor * σ_max

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
    split_factor::T = T(1.0)
) where {T<:Real, N}
    
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
        
        # Each component gets half the weight and same covariance
        comp1 = GaussianBelief(μ1, Σ)
        comp2 = GaussianBelief(μ2, Σ)
        w1 = w2 = weight / 2
        
    elseif method == :axis
        # Split along coordinate axis with maximum variance
        variances = [Σ[i,i] for i in 1:N]
        max_idx = argmax(variances)
        
        # Split distance
        σ = sqrt(variances[max_idx])
        δ = zeros(SVector{N, T})
        δ = setindex(δ, split_factor * σ, max_idx)
        
        # New means
        μ1 = μ + δ
        μ2 = μ - δ
        
        # Each component gets half the weight and same covariance
        comp1 = GaussianBelief(μ1, Σ)
        comp2 = GaussianBelief(μ2, Σ)
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
    eigvals = eigvals(Symmetric(comp.Σ))
    return maximum(eigvals) > threshold
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
