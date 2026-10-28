# Sigma-point transformation utilities

"""
    SigmaPoints{T, N}

Container for sigma points and their weights.

# Fields
- `points::Vector{SVector{N, T}}`: Sigma point locations
- `weights_mean::Vector{T}`: Weights for mean computation
- `weights_cov::Vector{T}`: Weights for covariance computation
"""
struct SigmaPoints{T, N}
    points::Vector{SVector{N, T}}
    weights_mean::Vector{T}
    weights_cov::Vector{T}
end

"""
    unscented_transform(μ, Σ; α=1e-3, β=2.0, κ=0.0)

Generate sigma points using the scaled unscented transform.

The scaled unscented transform creates 2n+1 sigma points that capture
the mean and covariance of a Gaussian distribution.

# Arguments
- `μ`: Mean vector (length n)
- `Σ`: Covariance matrix (n×n, SPD)
- `α`: Spread parameter (typically 1e-4 ≤ α ≤ 1)
- `β`: Distribution parameter (β=2 optimal for Gaussian)
- `κ`: Secondary scaling (typically 0 or 3-n)

# Returns
A `SigmaPoints` object containing 2n+1 sigma points and weights.

# References
- T. Humphreys, Statistical Estimation Theory course notes (UT Austin ASE 381P.6, 2023), Ch. 15
- Julier & Uhlmann (2004)
"""
function unscented_transform(μ::SVector{N, T}, Σ::SMatrix{N, N, T}; 
                             α=1e-3, β=2.0, κ=0.0) where {N, T<:Real}
    # Scaling parameters
    λ = α^2 * (N + κ) - N
    γ = sqrt(N + λ)
    
    # Compute matrix square root
    S = cholesky(Symmetric(Σ)).L
    
    # Generate sigma points
    points = Vector{SVector{N, T}}(undef, 2N + 1)
    points[1] = μ
    
    for i in 1:N
        points[i + 1] = μ + γ * S[:, i]
        points[N + i + 1] = μ - γ * S[:, i]
    end
    
    # Weights for mean
    w_m = Vector{T}(undef, 2N + 1)
    w_m[1] = λ / (N + λ)
    w_m[2:end] .= 1 / (2 * (N + λ))
    
    # Weights for covariance
    w_c = Vector{T}(undef, 2N + 1)
    w_c[1] = λ / (N + λ) + (1 - α^2 + β)
    w_c[2:end] .= 1 / (2 * (N + λ))
    
    return SigmaPoints{T, N}(points, w_m, w_c)
end

"""
    transform_sigma_points(f, sigma_points)

Apply transformation f to each sigma point.

# Arguments
- `f`: Transformation function (can be nonlinear)
- `sigma_points`: SigmaPoints object

# Returns
Transformed sigma points as a vector.
"""
function transform_sigma_points(f, sigma_points::SigmaPoints{T, N}) where {T, N}
    return [f(χ) for χ in sigma_points.points]
end

"""
    reconstruct_gaussian(transformed_points, weights_mean, weights_cov)

Reconstruct mean and covariance from transformed sigma points.

# Arguments
- `transformed_points`: Vector of transformed sigma points
- `weights_mean`: Weights for mean computation
- `weights_cov`: Weights for covariance computation

# Returns
A tuple `(μ, Σ)` of the reconstructed mean and covariance.
"""
function reconstruct_gaussian(transformed_points::Vector, weights_mean, weights_cov)
    # Compute mean
    μ = sum(w * y for (w, y) in zip(weights_mean, transformed_points))
    
    # Compute covariance
    n = length(μ)
    Σ = zeros(eltype(μ), n, n)
    
    for (w, y) in zip(weights_cov, transformed_points)
        dy = y - μ
        Σ += w * (dy * dy')
    end
    
    Σ = make_symmetric(Σ)
    return μ, Σ
end

"""
    cubature_transform(μ, Σ)

Generate sigma points using the spherical cubature rule.

The cubature Kalman filter uses 2n sigma points (no center point)
placed symmetrically about the mean.

# Arguments
- `μ`: Mean vector (length n)
- `Σ`: Covariance matrix (n×n, SPD)

# Returns
A `SigmaPoints` object containing 2n sigma points with equal weights.

# References
- R. Zanetti, "Quadrature Kalman Filter," ASE 381P.7 lecture slides (UT Austin, 2023)
- Arasaratnam & Haykin (2009)
"""
function cubature_transform(μ::SVector{N, T}, Σ::SMatrix{N, N, T}) where {N, T<:Real}
    # Matrix square root
    S = cholesky(Symmetric(Σ)).L
    
    # Generate 2n cubature points
    γ = sqrt(N)
    points = Vector{SVector{N, T}}(undef, 2N)
    
    for i in 1:N
        points[i] = μ + γ * S[:, i]
        points[N + i] = μ - γ * S[:, i]
    end
    
    # Equal weights
    w = fill(1 / (2N), 2N)
    
    return SigmaPoints{T, N}(points, w, w)
end

"""
    gauss_hermite_points(n, order=3)

Generate 1D Gauss-Hermite quadrature points and weights.

# Arguments
- `n`: Number of points (typically 3-5)
- `order`: Polynomial order (n points integrate polynomials up to order 2n-1)

# Returns
Tuple of (points, weights) for 1D quadrature.
"""
function gauss_hermite_points(n::Int)
    # Using standard lookup tables for common cases
    if n == 3
        ξ = [-sqrt(3/2), 0.0, sqrt(3/2)]
        w = [1/6, 2/3, 1/6]
    elseif n == 5
        ξ = [-sqrt(5 + 2*sqrt(10/7)), -sqrt(5 - 2*sqrt(10/7)), 0.0,
             sqrt(5 - 2*sqrt(10/7)), sqrt(5 + 2*sqrt(10/7))]
        w = [1/(12*sqrt(10/7)), (49 + 17*sqrt(10/7))/180, 64/180,
             (49 + 17*sqrt(10/7))/180, 1/(12*sqrt(10/7))]
    else
        error("Gauss-Hermite points only implemented for n=3 or n=5")
    end
    
    return ξ, w
end
