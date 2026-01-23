"""
Component merging and pruning utilities for Gaussian mixture management.

Merging combines similar components to reduce computational cost while
maintaining approximation quality.
"""

"""
    mahalanobis_distance(μ1, Σ1, μ2, Σ2)

Compute the Mahalanobis distance between two Gaussian distributions:

```
d² = (μ1 - μ2)ᵀ Σ⁻¹ (μ1 - μ2)
```

where Σ = (Σ1 + Σ2)/2 is the average covariance.

# Returns
- Distance (non-negative scalar)
"""
function mahalanobis_distance(
    μ1::AbstractVector{T},
    Σ1::AbstractMatrix{T},
    μ2::AbstractVector{T},
    Σ2::AbstractMatrix{T}
) where T<:Real
    
    Δμ = μ1 - μ2
    Σ_avg = (Σ1 + Σ2) / 2
    
    # Ensure positive definite
    Σ_avg = ensure_spd(Σ_avg)
    
    # Compute distance
    d_squared = Δμ' * (Σ_avg \ Δμ)
    return sqrt(max(zero(T), d_squared))
end

"""
    bhattacharyya_distance(μ1, Σ1, μ2, Σ2)

Compute the Bhattacharyya distance between two Gaussian distributions:

```
d_B = (1/8)(μ1-μ2)ᵀΣ⁻¹(μ1-μ2) + (1/2)log(|Σ|/√(|Σ1||Σ2|))
```

where Σ = (Σ1 + Σ2)/2.

# Returns
- Distance (non-negative scalar)
"""
function bhattacharyya_distance(
    μ1::AbstractVector{T},
    Σ1::AbstractMatrix{T},
    μ2::AbstractVector{T},
    Σ2::AbstractMatrix{T}
) where T<:Real
    
    Δμ = μ1 - μ2
    Σ_avg = (Σ1 + Σ2) / 2
    
    # Ensure positive definite
    Σ_avg = ensure_spd(Σ_avg)
    
    # Mahalanobis term
    d_mahal = Δμ' * (Σ_avg \ Δμ)
    
    # Determinant term
    det1 = det(Σ1)
    det2 = det(Σ2)
    det_avg = det(Σ_avg)
    
    d_det = log(det_avg / sqrt(det1 * det2))
    
    return (1/8) * d_mahal + (1/2) * d_det
end

"""
    merge_components(μ1, Σ1, w1, μ2, Σ2, w2)

Merge two Gaussian components into a single component using the moment-matching formula:

```
w_merged = w1 + w2
μ_merged = (w1*μ1 + w2*μ2) / w_merged
Σ_merged = (w1*(Σ1 + μ1*μ1ᵀ) + w2*(Σ2 + μ2*μ2ᵀ)) / w_merged - μ_merged*μ_mergedᵀ
```

# Returns
- `(μ_merged, Σ_merged, w_merged)`
"""
function merge_components(
    μ1::SVector{N, T},
    Σ1::SMatrix{N, N, T},
    w1::T,
    μ2::SVector{N, T},
    Σ2::SMatrix{N, N, T},
    w2::T
) where {T<:Real, N}
    
    w_total = w1 + w2
    
    # Merged mean (weighted average)
    μ_merged = (w1 * μ1 + w2 * μ2) / w_total
    
    # Merged covariance (moment matching)
    # Σ = E[xxᵀ] - μμᵀ
    # E[xxᵀ] = w1*(Σ1 + μ1μ1ᵀ) + w2*(Σ2 + μ2μ2ᵀ)
    M1 = Σ1 + μ1 * μ1'
    M2 = Σ2 + μ2 * μ2'
    M_merged = (w1 * M1 + w2 * M2) / w_total
    
    Σ_merged = M_merged - μ_merged * μ_merged'
    
    # Ensure positive definite
    Σ_merged = ensure_spd(Σ_merged)
    
    return μ_merged, Σ_merged, w_total
end

"""
    merge_components(comp1::GaussianBelief, w1, comp2::GaussianBelief, w2)

Merge two Gaussian belief components.

# Returns
- `(merged_belief, merged_weight)`
"""
function merge_components(
    comp1::GaussianBelief{T, N},
    w1::T,
    comp2::GaussianBelief{T, N},
    w2::T
) where {T<:Real, N}
    
    μ, Σ, w = merge_components(comp1.μ, comp1.Σ, w1, comp2.μ, comp2.Σ, w2)
    return GaussianBelief(μ, Σ), w
end

"""
    merge_components(ekf1::ExtendedKalmanFilter, w1, ekf2::ExtendedKalmanFilter, w2)

Merge two EKF components for use in Gaussian Sum Filter.

# Returns
- `(merged_ekf, merged_weight)`
"""
function merge_components(
    ekf1::ExtendedKalmanFilter{T, N},
    w1::T,
    ekf2::ExtendedKalmanFilter{T, N},
    w2::T
) where {T<:Real, N}
    
    μ1 = state_estimate(ekf1)
    Σ1 = covariance(ekf1)
    μ2 = state_estimate(ekf2)
    Σ2 = covariance(ekf2)
    
    μ, Σ, w = merge_components(μ1, Σ1, w1, μ2, Σ2, w2)
    
    return ExtendedKalmanFilter(μ, Σ), w
end

"""
    find_closest_components(gsf::GaussianSumFilter; metric=:mahalanobis)

Find the pair of components with minimum distance.

# Keyword Arguments
- `metric`: Distance metric to use
  - `:mahalanobis` - Mahalanobis distance (default)
  - `:bhattacharyya` - Bhattacharyya distance

# Returns
- `(i, j, distance)`: Indices of closest pair and their distance
"""
function find_closest_components(
    gsf::GaussianSumFilter{T, N};
    metric::Symbol = :mahalanobis
) where {T<:Real, N}
    
    M = num_components(gsf)
    if M < 2
        return nothing
    end
    
    min_distance = T(Inf)
    min_i = 1
    min_j = 2
    
    # Compute all pairwise distances
    for i in 1:(M-1)
        μi = state_estimate(gsf.components[i])
        Σi = covariance(gsf.components[i])
        
        for j in (i+1):M
            μj = state_estimate(gsf.components[j])
            Σj = covariance(gsf.components[j])
            
            if metric == :mahalanobis
                d = mahalanobis_distance(μi, Σi, μj, Σj)
            elseif metric == :bhattacharyya
                d = bhattacharyya_distance(μi, Σi, μj, Σj)
            else
                error("Unknown metric: $metric")
            end
            
            if d < min_distance
                min_distance = d
                min_i = i
                min_j = j
            end
        end
    end
    
    return (min_i, min_j, min_distance)
end

"""
    merge_closest_components!(gsf::GaussianSumFilter; kwargs...)

Merge the two closest components in the mixture.

# Keyword Arguments
- `metric`: Distance metric (`:mahalanobis` or `:bhattacharyya`)

# Returns
- `true` if merge was performed, `false` if not enough components
"""
function merge_closest_components!(
    gsf::GaussianSumFilter{T, N};
    kwargs...
) where {T<:Real, N}
    
    result = find_closest_components(gsf; kwargs...)
    if result === nothing
        return false
    end
    
    i, j, _ = result
    
    # Merge components i and j
    merged_ekf, merged_weight = merge_components(
        gsf.components[i], gsf.weights[i],
        gsf.components[j], gsf.weights[j]
    )
    
    # Remove components i and j, add merged
    keep_idx = setdiff(1:num_components(gsf), [i, j])
    gsf.components = [gsf.components[keep_idx]; merged_ekf]
    gsf.weights = [gsf.weights[keep_idx]; merged_weight]
    
    return true
end

"""
    merge_similar_components!(gsf::GaussianSumFilter, threshold; kwargs...)

Merge all component pairs with distance below threshold.

# Arguments
- `gsf`: Gaussian Sum Filter to modify
- `threshold`: Maximum distance for merging

# Keyword Arguments
- `metric`: Distance metric (`:mahalanobis` or `:bhattacharyya`)

# Example
```julia
# Merge components within Mahalanobis distance of 2.0
merge_similar_components!(gsf, 2.0)
```
"""
function merge_similar_components!(
    gsf::GaussianSumFilter{T, N},
    threshold::T;
    metric::Symbol = :mahalanobis
) where {T<:Real, N}
    
    merged_count = 0
    
    # Keep merging until no more pairs are below threshold
    while num_components(gsf) > 1
        result = find_closest_components(gsf; metric=metric)
        if result === nothing
            break
        end
        
        i, j, distance = result
        
        if distance <= threshold
            # Merge this pair
            merged_ekf, merged_weight = merge_components(
                gsf.components[i], gsf.weights[i],
                gsf.components[j], gsf.weights[j]
            )
            
            # Remove components i and j, add merged
            keep_idx = setdiff(1:num_components(gsf), [i, j])
            gsf.components = [gsf.components[keep_idx]; merged_ekf]
            gsf.weights = [gsf.weights[keep_idx]; merged_weight]
            
            merged_count += 1
        else
            # No more pairs below threshold
            break
        end
    end
    
    return merged_count
end

"""
    reduce_to_n_components!(gsf::GaussianSumFilter, n_target; kwargs...)

Reduce mixture to target number of components by iteratively merging closest pairs.

# Arguments
- `gsf`: Gaussian Sum Filter to modify
- `n_target`: Target number of components

# Keyword Arguments
- `metric`: Distance metric (`:mahalanobis` or `:bhattacharyya`)
"""
function reduce_to_n_components!(
    gsf::GaussianSumFilter,
    n_target::Int;
    kwargs...
)
    while num_components(gsf) > n_target
        success = merge_closest_components!(gsf; kwargs...)
        if !success
            break
        end
    end
    
    return gsf
end
