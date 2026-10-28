"""
    PointMassFilter{T<:Real, N}

Point Mass Filter (PMF) - Grid-based Bayesian filter for low-dimensional nonlinear systems.

The PMF discretizes the state space onto a fixed grid and represents the probability
density function directly at each grid point. This allows exact (up to discretization)
representation of arbitrary non-Gaussian distributions.

# Mathematical Formulation

State space is discretized: xᵢ ∈ {x₁, x₂, ..., xₘ}

Probability mass at grid point i: pᵢ = P(x = xᵢ)

Prediction (Chapman-Kolmogorov):
```math
p̄ᵢ(k+1) = ∑ⱼ P(x_{k+1} = xᵢ | x_k = xⱼ) pⱼ(k)
```

Update (Bayes' rule):
```math
pᵢ(k+1) = L(yₖ₊₁ | xᵢ) p̄ᵢ(k+1) / ∑ⱼ L(yₖ₊₁ | xⱼ) p̄ⱼ(k+1)
```

where L(y|x) is the measurement likelihood.

# Advantages
- No distributional assumptions
- Handles multi-modal densities naturally
- Exact Bayesian recursion (up to grid discretization)

# Limitations
- Curse of dimensionality (exponential grid growth)
- Practical only for n ≤ 3 dimensions
- Fixed grid may miss important regions

# Reference
- R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 3: Nonlinear Estimation (UT Austin ASE 381P.7, 2023), Ch. 4
- Bucy & Senne (1971)
- Sorenson & Alspach (1971)

# Fields
- `grid_points::Vector{SVector{N,T}}` - Grid point locations
- `probabilities::Vector{T}` - Probability mass at each point
- `bounds::Vector{Tuple{T,T}}` - State space bounds for each dimension
- `n_points_per_dim::Vector{Int}` - Number of grid points per dimension
"""
mutable struct PointMassFilter{T<:Real, N} <: AbstractFilter
    grid_points::Vector{SVector{N, T}}
    probabilities::Vector{T}
    bounds::Vector{Tuple{T, T}}  # (min, max) for each dimension
    n_points_per_dim::Vector{Int}
    
    # Storage
    predicted_probabilities::Vector{T}
end

"""
    PointMassFilter(bounds, n_points_per_dim, x0)

Create a Point Mass Filter with uniform grid.

# Arguments
- `bounds::Vector{Tuple{T,T}}` - State bounds [(xmin, xmax), (ymin, ymax), ...]
- `n_points_per_dim::Vector{Int}` - Number of grid points in each dimension
- `x0::SVector{N,T}` - Initial state (optional, defaults to center)
"""
function PointMassFilter(
    bounds::Vector{Tuple{T, T}},
    n_points_per_dim::Vector{Int},
    x0::SVector{N, T}
) where {T<:Real, N}
    
    @assert length(bounds) == N "Bounds must match state dimension"
    @assert length(n_points_per_dim) == N "Grid points spec must match state dimension"
    @assert all(n_points_per_dim .> 1) "Need at least 2 points per dimension"
    
    # Generate uniform grid
    grid_points = generate_uniform_grid(bounds, n_points_per_dim)
    n_total = length(grid_points)
    
    # Initialize probabilities
    probabilities = zeros(T, n_total)
    
    # Initialize with probability mass at closest grid point
    closest_idx = find_closest_grid_point(x0, grid_points)
    probabilities[closest_idx] = T(1.0)
    
    predicted_probabilities = copy(probabilities)
    
    return PointMassFilter{T, N}(
        grid_points, probabilities, bounds, n_points_per_dim, 
        predicted_probabilities
    )
end

function PointMassFilter(
    bounds::Vector{Tuple{T, T}},
    n_points_per_dim::Vector{Int},
    x0::Nothing = nothing
) where {T<:Real}
    
    N = length(bounds)
    @assert length(n_points_per_dim) == N "Grid points spec must match state dimension"
    @assert all(n_points_per_dim .> 1) "Need at least 2 points per dimension"
    
    # Generate uniform grid
    grid_points = generate_uniform_grid(bounds, n_points_per_dim)
    n_total = length(grid_points)
    
    # Initialize probabilities - Uniform initialization
    probabilities = zeros(T, n_total)
    probabilities .= T(1.0) / n_total
    
    predicted_probabilities = copy(probabilities)
    
    return PointMassFilter{T, N}(
        grid_points, probabilities, bounds, n_points_per_dim, 
        predicted_probabilities
    )
end

"""
    generate_uniform_grid(bounds, n_points_per_dim)

Generate uniform Cartesian grid over the specified bounds.
"""
function generate_uniform_grid(
    bounds::Vector{Tuple{T, T}},
    n_points_per_dim::Vector{Int}
) where {T<:Real}
    
    N = length(bounds)
    
    # Create 1D grids for each dimension
    grids_1d = [range(bounds[i][1], bounds[i][2], length=n_points_per_dim[i]) 
                for i in 1:N]
    
    # Generate Cartesian product
    grid_points = SVector{N, T}[]
    
    # Recursively generate all combinations
    function add_points!(coords, dim)
        if dim > N
            push!(grid_points, SVector{N, T}(coords...))
        else
            for val in grids_1d[dim]
                add_points!(vcat(coords, val), dim + 1)
            end
        end
    end
    
    add_points!(T[], 1)
    
    return grid_points
end

"""
    find_closest_grid_point(x, grid_points)

Find index of grid point closest to x.
"""
function find_closest_grid_point(x::SVector{N, T}, grid_points::Vector{SVector{N, T}}) where {N, T}
    min_dist = Inf
    closest_idx = 1
    
    for (i, xg) in enumerate(grid_points)
        dist = norm(x - xg)
        if dist < min_dist
            min_dist = dist
            closest_idx = i
        end
    end
    
    return closest_idx
end

"""
    predict!(pmf::PointMassFilter, model_dyn, u, dt)

Prediction step using transition probabilities.
"""
function predict!(
    pmf::PointMassFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    u = nothing,
    dt::Real = 1.0
) where {T, N}
    
    n_grid = length(pmf.grid_points)
    pmf.predicted_probabilities .= zero(T)
    
    # For each current grid point
    for j in 1:n_grid
        if pmf.probabilities[j] < eps(T)
            continue  # Skip negligible probabilities
        end
        
        # Propagate through dynamics
        xj = pmf.grid_points[j]
        x_prop = propagate(model_dyn, xj, u, dt)
        
        # For Gaussian process noise, compute transition probability to each grid point
        Q = process_noise_cov(model_dyn, dt)
        
        for i in 1:n_grid
            xi = pmf.grid_points[i]
            
            # Transition probability: P(x_{k+1} = xi | x_k = xj)
            # Assuming Gaussian process noise: x_{k+1} = f(x_k) + v, v ~ N(0, Q)
            # P(xi | xj) ≈ N(xi; f(xj), Q)
            diff = xi - x_prop
            
            # Gaussian likelihood
            if N == 1
                trans_prob = exp(-0.5 * diff[1]^2 / Q[1,1]) / sqrt(2π * Q[1,1])
            else
                # Multivariate Gaussian
                dist = MvNormal(x_prop, Q)
                trans_prob = pdf(dist, xi)
            end
            
            pmf.predicted_probabilities[i] += trans_prob * pmf.probabilities[j]
        end
    end
    
    # Normalize
    total_prob = sum(pmf.predicted_probabilities)
    if total_prob > eps(T)
        pmf.predicted_probabilities ./= total_prob
    else
        # Uniform fallback if all probabilities collapsed
        pmf.predicted_probabilities .= T(1.0) / n_grid
    end
    
    return pmf
end

"""
    update!(pmf::PointMassFilter, model_meas, y)

Measurement update using Bayes' rule.
"""
function update!(
    pmf::PointMassFilter{T, N},
    model_meas::AbstractMeasurementModel,
    y::AbstractVector
) where {T, N}
    
    n_grid = length(pmf.grid_points)
    R = measurement_noise_cov(model_meas)
    
    # Compute likelihood at each grid point
    likelihoods = zeros(T, n_grid)
    
    for i in 1:n_grid
        xi = pmf.grid_points[i]
        ŷ = measure(model_meas, xi)
        
        # Measurement likelihood: L(y | xi) = N(y; h(xi), R)
        innov = y - ŷ
        
        if length(y) == 1
            likelihoods[i] = exp(-0.5 * innov[1]^2 / R[1,1])
        else
            # Multivariate
            likelihoods[i] = exp(-0.5 * dot(innov, R \ innov))
        end
    end
    
    # Bayes' rule: p(x|y) ∝ L(y|x) * p̄(x)
    pmf.probabilities .= likelihoods .* pmf.predicted_probabilities
    
    # Normalize
    total_prob = sum(pmf.probabilities)
    if total_prob > eps(T)
        pmf.probabilities ./= total_prob
    else
        # If all likelihoods are zero, keep prior
        pmf.probabilities .= pmf.predicted_probabilities
    end
    
    return pmf
end

"""
    step!(pmf, model_dyn, model_meas, u, y, dt)

Combined predict and update step.
"""
function step!(
    pmf::PointMassFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    model_meas::AbstractMeasurementModel,
    u::AbstractVector,
    y::AbstractVector,
    dt::T = T(1.0)
) where {T, N}
    
    predict!(pmf, model_dyn, u, dt)
    update!(pmf, model_meas, y)
    return pmf
end

"""
    state_estimate(pmf::PointMassFilter)

Compute MMSE estimate (mean of discrete distribution).
"""
function state_estimate(pmf::PointMassFilter{T, N}) where {T, N}
    x̂ = zeros(SVector{N, T})
    for (i, xi) in enumerate(pmf.grid_points)
        x̂ += pmf.probabilities[i] * xi
    end
    return x̂
end

"""
    covariance(pmf::PointMassFilter)

Compute covariance of discrete distribution.
"""
function covariance(pmf::PointMassFilter{T, N}) where {T, N}
    x̂ = state_estimate(pmf)
    P = zeros(SMatrix{N, N, T})
    
    for (i, xi) in enumerate(pmf.grid_points)
        diff = xi - x̂
        P += pmf.probabilities[i] * (diff * diff')
    end
    
    return make_symmetric(P)
end

"""
    get_probability_mass(pmf::PointMassFilter)

Return grid points and their probabilities.
"""
function get_probability_mass(pmf::PointMassFilter)
    return pmf.grid_points, pmf.probabilities
end

"""
    maximum_probability_state(pmf::PointMassFilter)

Return the grid point with maximum probability (MAP estimate).
"""
function maximum_probability_state(pmf::PointMassFilter)
    max_idx = argmax(pmf.probabilities)
    return pmf.grid_points[max_idx]
end
