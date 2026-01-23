"""
Covariance localization utilities for ensemble methods.

Localization reduces spurious long-range correlations in ensemble covariance
estimates caused by finite ensemble size. This is critical for high-dimensional
problems (atmosphere, ocean models).

# Theory

The localized covariance is computed as:
```
P_loc = ρ ⊙ P
```

where ⊙ is element-wise (Schur) product and ρ is a correlation function
that decays with distance.

# Common Localization Functions
- Gaspari-Cohn (GC): Smooth compact support, C2 continuous
- Gaussian: exp(-d²/(2r²))
- Boxcar: Binary cutoff
"""

"""
    gaspari_cohn(distance, radius)

Gaspari-Cohn localization function with compact support.

This is a fifth-order piecewise polynomial that:
- ρ(0) = 1
- ρ(r) = 0 for d ≥ 2r
- Is C2 continuous (smooth)

# Arguments
- `distance`: Distance between locations
- `radius`: Localization radius (half-width of support)

# Returns
- Correlation value in [0, 1]

# Reference
Gaspari & Cohn (1999), "Construction of correlation functions in two and three dimensions"
"""
function gaspari_cohn(distance::T, radius::T) where T<:Real
    if radius <= 0
        return one(T)
    end
    
    z = abs(distance) / radius
    
    if z >= 2
        return zero(T)
    elseif z >= 1
        # Far branch: [r, 2r]
        z2 = z^2
        z3 = z^3
        return (1/12) * z^5 - (1/2) * z^4 + (5/8) * z3 + (5/3) * z2 - 5*z + 4 - (2/3) / z
    else
        # Near branch: [0, r]
        z2 = z^2
        z3 = z^3
        return 1 - (5/3) * z2 + (5/8) * z3 + (1/2) * z^4 - (1/4) * z^5
    end
end

"""
    gaussian_localization(distance, radius)

Gaussian localization function (no compact support).

```
ρ(d) = exp(-d²/(2r²))
```

# Arguments
- `distance`: Distance between locations
- `radius`: Localization radius (standard deviation)

# Returns
- Correlation value in (0, 1]
"""
function gaussian_localization(distance::T, radius::T) where T<:Real
    if radius <= 0
        return one(T)
    end
    
    return exp(-(distance^2) / (2 * radius^2))
end

"""
    boxcar_localization(distance, radius)

Boxcar (binary cutoff) localization function.

```
ρ(d) = 1 if d < r, else 0
```

# Arguments
- `distance`: Distance between locations
- `radius`: Localization radius (cutoff distance)

# Returns
- Correlation value in {0, 1}
"""
function boxcar_localization(distance::T, radius::T) where T<:Real
    return distance < radius ? one(T) : zero(T)
end

"""
    localization_matrix(positions, radius; func=gaspari_cohn)

Construct full localization matrix from spatial positions.

# Arguments
- `positions`: Vector of position vectors (e.g., [x1, x2, ...])
- `radius`: Localization radius

# Keyword Arguments
- `func`: Localization function (gaspari_cohn, gaussian_localization, or custom)
- `metric`: Distance metric (default: Euclidean)

# Returns
- Symmetric localization matrix ρ[i,j] = func(||xi - xj||, radius)

# Example
```julia
# 1D positions
positions = [SVector(x) for x in 0:0.1:10]
ρ = localization_matrix(positions, 2.0)
```
"""
function localization_matrix(
    positions::Vector{<:AbstractVector{T}},
    radius::T;
    func::Function = gaspari_cohn
) where T<:Real
    
    n = length(positions)
    ρ = Matrix{T}(undef, n, n)
    
    for i in 1:n
        ρ[i, i] = one(T)  # Self-correlation is 1
        
        for j in (i+1):n
            # Euclidean distance
            d = sqrt(sum(abs2, positions[i] - positions[j]))
            ρ[i, j] = func(d, radius)
            ρ[j, i] = ρ[i, j]  # Symmetric
        end
    end
    
    return ρ
end

"""
    apply_localization(P, ρ)

Apply localization to covariance matrix using Schur product.

```
P_loc = ρ ⊙ P
```

# Arguments
- `P`: Covariance matrix to localize
- `ρ`: Localization matrix

# Returns
- Localized covariance matrix
"""
function apply_localization(
    P::AbstractMatrix{T},
    ρ::AbstractMatrix{T}
) where T<:Real
    
    @assert size(P) == size(ρ) "Matrices must have same size"
    return P .* ρ
end

"""
    localize_ensemble_covariance!(enkf, positions, radius; func=gaspari_cohn)

Apply localization to ensemble covariance in-place.

This modifies the ensemble to produce a localized covariance estimate.

# Arguments
- `enkf`: Ensemble filter (EnKF or ETKF)
- `positions`: Spatial positions for each state variable
- `radius`: Localization radius

# Keyword Arguments
- `func`: Localization function

# Note
This is a simplified localization. For large systems, use local analysis
or domain decomposition for efficiency.
"""
function localize_ensemble_covariance!(
    enkf::Union{EnsembleKalmanFilter, EnsembleTransformKalmanFilter},
    positions::Vector{<:AbstractVector{T}},
    radius::T;
    func::Function = gaspari_cohn
) where T<:Real
    
    # Compute localization matrix
    ρ = localization_matrix(positions, radius; func=func)
    
    # Get ensemble covariance
    P = Matrix(covariance(enkf))
    
    # Apply localization
    P_loc = apply_localization(P, ρ)
    
    # Reconstruct ensemble with localized covariance
    # This is non-trivial; one approach is to use square-root adjustment
    x̄ = Vector(state_estimate(enkf))
    
    # Compute square root of localized covariance
    P_loc_sqrt = Matrix(cholesky(Symmetric(P_loc)).L)
    
    # Resample ensemble
    n_ens = num_ensemble(enkf)
    for i in 1:n_ens
        # Sample from N(x̄, P_loc)
        z = randn(T, length(x̄))
        enkf.ensemble[:, i] = x̄ + P_loc_sqrt * z
    end
    
    return enkf
end

"""
    distance_matrix(positions)

Compute pairwise distance matrix for positions.

# Arguments
- `positions`: Vector of position vectors

# Returns
- Matrix D where D[i,j] = ||positions[i] - positions[j]||
"""
function distance_matrix(positions::Vector{<:AbstractVector{T}}) where T<:Real
    n = length(positions)
    D = Matrix{T}(undef, n, n)
    
    for i in 1:n
        D[i, i] = zero(T)
        for j in (i+1):n
            d = sqrt(sum(abs2, positions[i] - positions[j]))
            D[i, j] = d
            D[j, i] = d
        end
    end
    
    return D
end

"""
    adaptive_localization_radius(ensemble_spread, target_rank; factor=1.0)

Compute adaptive localization radius based on ensemble spread.

Larger spread suggests better sampling, allowing larger radius.

# Arguments
- `ensemble_spread`: RMS ensemble spread
- `target_rank`: Target effective rank for localization
- `factor`: Scaling factor (default: 1.0)

# Returns
- Localization radius
"""
function adaptive_localization_radius(
    ensemble_spread::T,
    target_rank::Int;
    factor::T = T(1.0)
) where T<:Real
    
    # Heuristic: radius ~ spread * sqrt(target_rank)
    return factor * ensemble_spread * sqrt(target_rank)
end
