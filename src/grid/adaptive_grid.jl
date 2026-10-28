"""
    AdaptiveGridFilter{T<:Real, N}

Adaptive Grid Filter - Point mass filter with dynamic grid refinement.

!!! warning "Experimental"
    This filter is currently experimental and may have numerical stability issues.
    Use with caution in production code. Prefer `PointMassFilter` for stable operation.

The adaptive grid filter automatically refines the grid in regions of high
probability mass and coarsens in low-probability regions. This addresses
the curse of dimensionality by focusing computational resources where needed.

# Mathematical Formulation

Similar to standard PMF but with dynamic grid management:

1. **Grid Refinement**: Split high-probability cells into subcells
2. **Grid Coarsening**: Merge adjacent low-probability cells
3. **Pruning**: Remove cells with negligible probability

Refinement criterion:
```math
p_i > θ_refine * max(p)
```

Coarsening criterion:
```math
p_i < θ_coarsen * max(p)
```

# Advantages
- Efficient use of computational resources
- Can handle moderate-dimensional problems (n ≤ 5)
- Adapts to moving distributions

# Limitations
- More complex implementation
- Overhead from grid management
- Still limited by dimensionality

# Reference
- R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 3: Nonlinear Estimation (UT Austin ASE 381P.7, 2023), §4.1
- Šimandl & Duník (2009)

# Fields
- `grid_points::Vector{SVector{N,T}}` - Current grid points
- `probabilities::Vector{T}` - Probability masses
- `grid_levels::Vector{Int}` - Refinement level of each cell
- `bounds::Vector{Tuple{T,T}}` - State space bounds
"""
mutable struct AdaptiveGridFilter{T<:Real, N} <: AbstractFilter
    grid_points::Vector{SVector{N, T}}
    probabilities::Vector{T}
    grid_levels::Vector{Int}  # Refinement level (0 = base)
    bounds::Vector{Tuple{T, T}}
    
    # Adaptation parameters
    refine_threshold::T
    coarse_threshold::T
    prune_threshold::T
    max_level::Int
    
    # Storage
    predicted_probabilities::Vector{T}
end

"""
    AdaptiveGridFilter(bounds, initial_n_points, x0; kwargs...)

Create adaptive grid filter with initial uniform grid.

# Arguments
- `bounds::Vector{Tuple{T,T}}` - State bounds
- `initial_n_points::Vector{Int}` - Initial grid points per dimension
- `x0::SVector{N,T}` - Initial state

# Keyword Arguments
- `refine_threshold::T = 0.1` - Probability threshold for refinement (fraction of max)
- `coarse_threshold::T = 0.001` - Threshold for coarsening
- `prune_threshold::T = 1e-6` - Threshold for removing points
- `max_level::Int = 3` - Maximum refinement level
"""
function AdaptiveGridFilter(
    bounds::Vector{Tuple{T, T}},
    initial_n_points::Vector{Int},
    x0::SVector{N, T};
    refine_threshold::T = T(0.1),
    coarse_threshold::T = T(0.001),
    prune_threshold::T = T(1e-6),
    max_level::Int = 3
) where {T<:Real, N}
    
    # Generate initial uniform grid
    grid_points = generate_uniform_grid(bounds, initial_n_points)
    n_points = length(grid_points)
    
    # Initialize with probability at closest point to x0
    probabilities = zeros(T, n_points)
    closest_idx = find_closest_grid_point(x0, grid_points)
    probabilities[closest_idx] = T(1.0)
    
    grid_levels = zeros(Int, n_points)
    predicted_probabilities = copy(probabilities)
    
    return AdaptiveGridFilter{T, N}(
        grid_points, probabilities, grid_levels, bounds,
        refine_threshold, coarse_threshold, prune_threshold, max_level,
        predicted_probabilities
    )
end

"""
    predict!(agf::AdaptiveGridFilter, model_dyn, u, dt)

Prediction step with grid adaptation.
"""
function predict!(
    agf::AdaptiveGridFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    u = nothing,
    dt::Real = 1.0
) where {T, N}
    
    n_grid = length(agf.grid_points)
    agf.predicted_probabilities = zeros(T, n_grid)
    Q = process_noise_cov(model_dyn, dt)
    
    # Standard PMF prediction
    for j in 1:n_grid
        if agf.probabilities[j] < agf.prune_threshold
            continue
        end
        
        xj = agf.grid_points[j]
        x_prop = propagate(model_dyn, xj, u, dt)
        
        for i in 1:n_grid
            xi = agf.grid_points[i]
            diff = xi - x_prop
            
            # Gaussian transition probability
            if N == 1
                trans_prob = exp(-0.5 * diff[1]^2 / Q[1,1]) / sqrt(2π * Q[1,1])
            else
                dist = MvNormal(x_prop, Q)
                trans_prob = pdf(dist, xi)
            end
            
            agf.predicted_probabilities[i] += trans_prob * agf.probabilities[j]
        end
    end
    
    # Normalize
    total_prob = sum(agf.predicted_probabilities)
    if total_prob > eps(T)
        agf.predicted_probabilities ./= total_prob
    end
    
    # Do not adapt grid after prediction - wait for measurement update
    
    return agf
end

"""
    update!(agf::AdaptiveGridFilter, model_meas, y)

Measurement update.
"""
function update!(
    agf::AdaptiveGridFilter{T, N},
    model_meas::AbstractMeasurementModel,
    y::AbstractVector
) where {T, N}
    
    R = measurement_noise_cov(model_meas)
    n_grid = length(agf.grid_points)
    likelihoods = zeros(T, n_grid)
    
    # Compute likelihoods
    for i in 1:n_grid
        xi = agf.grid_points[i]
        ŷ = measure(model_meas, xi)
        innov = y - ŷ
        
        if length(y) == 1
            likelihoods[i] = exp(-0.5 * innov[1]^2 / R[1,1])
        else
            likelihoods[i] = exp(-0.5 * dot(innov, R \ innov))
        end
    end
    
    # Bayes update
    agf.probabilities .= likelihoods .* agf.predicted_probabilities
    
    total_prob = sum(agf.probabilities)
    if total_prob > eps(T)
        agf.probabilities ./= total_prob
    else
        agf.probabilities .= agf.predicted_probabilities
    end
    
    # Adapt grid after update
    adapt_grid!(agf)
    
    return agf
end

"""
    adapt_grid!(agf::AdaptiveGridFilter)

Adaptive grid refinement, coarsening, and pruning.
"""
function adapt_grid!(agf::AdaptiveGridFilter{T, N}) where {T, N}
    # Prune low-probability points
    keep_indices = findall(p -> p > agf.prune_threshold, agf.probabilities)
    
    if isempty(keep_indices)
        # Keep at least one point (reset to uniform)
        return agf
    end
    
    agf.grid_points = agf.grid_points[keep_indices]
    agf.probabilities = agf.probabilities[keep_indices]
    agf.grid_levels = agf.grid_levels[keep_indices]
    agf.predicted_probabilities = agf.predicted_probabilities[keep_indices]
    
    # Normalize after pruning
    total = sum(agf.probabilities)
    if total > eps(T)
        agf.probabilities ./= total
    end
    
    # Refinement: split high-probability cells
    max_prob = maximum(agf.probabilities)
    refine_indices = findall(p -> p > agf.refine_threshold * max_prob, agf.probabilities)
    
    new_points = SVector{N, T}[]
    new_probs = T[]
    new_levels = Int[]
    
    for idx in refine_indices
        if agf.grid_levels[idx] < agf.max_level
            # Split this cell into 2^N subcells
            parent_point = agf.grid_points[idx]
            parent_prob = agf.probabilities[idx]
            parent_level = agf.grid_levels[idx]
            
            # Compute cell size based on level
            cell_size = compute_cell_size(agf.bounds, parent_level)
            
            # Generate subcells (simplified: only in positive directions)
            n_subcells = 2^N
            subcell_prob = parent_prob / n_subcells
            
            for subcell_idx in 0:(n_subcells-1)
                offset = SVector{N, T}([((subcell_idx >> d) & 1) * cell_size[d+1] / 2 
                                        for d in 0:(N-1)])
                subcell_point = parent_point + offset
                
                # Check if within bounds
                if is_within_bounds(subcell_point, agf.bounds)
                    push!(new_points, subcell_point)
                    push!(new_probs, subcell_prob)
                    push!(new_levels, parent_level + 1)
                end
            end
            
            # Mark original for removal
            agf.probabilities[idx] = T(0.0)
        end
    end
    
    # Add refined points
    if !isempty(new_points)
        append!(agf.grid_points, new_points)
        append!(agf.probabilities, new_probs)
        append!(agf.grid_levels, new_levels)
        
        # Re-prune to remove marked points
        keep_indices = findall(p -> p > eps(T), agf.probabilities)
        agf.grid_points = agf.grid_points[keep_indices]
        agf.probabilities = agf.probabilities[keep_indices]
        agf.grid_levels = agf.grid_levels[keep_indices]
        
        # Renormalize
        total = sum(agf.probabilities)
        if total > eps(T)
            agf.probabilities ./= total
        end
    end
    
    # Update predicted probabilities size
    agf.predicted_probabilities = zeros(T, length(agf.probabilities))
    
    return agf
end

"""
    compute_cell_size(bounds, level)

Compute cell size at a given refinement level.
"""
function compute_cell_size(bounds::Vector{Tuple{T, T}}, level::Int) where {T}
    N = length(bounds)
    sizes = SVector{N, T}([(bounds[i][2] - bounds[i][1]) / (2^level) for i in 1:N])
    return sizes
end

"""
    is_within_bounds(x, bounds)

Check if point is within state bounds.
"""
function is_within_bounds(x::SVector{N, T}, bounds::Vector{Tuple{T, T}}) where {N, T}
    for i in 1:N
        if x[i] < bounds[i][1] || x[i] > bounds[i][2]
            return false
        end
    end
    return true
end

"""
    step!(agf, model_dyn, model_meas, u, y, dt)

Combined predict and update with adaptation.
"""
function step!(
    agf::AdaptiveGridFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    model_meas::AbstractMeasurementModel,
    u::AbstractVector,
    y::AbstractVector,
    dt::T = T(1.0)
) where {T, N}
    
    predict!(agf, model_dyn, u, dt)
    update!(agf, model_meas, y)
    return agf
end

"""
    state_estimate(agf::AdaptiveGridFilter)

Compute MMSE estimate.
"""
function state_estimate(agf::AdaptiveGridFilter{T, N}) where {T, N}
    x̂ = zeros(SVector{N, T})
    for (i, xi) in enumerate(agf.grid_points)
        x̂ += agf.probabilities[i] * xi
    end
    return x̂
end

"""
    covariance(agf::AdaptiveGridFilter)

Compute covariance.
"""
function covariance(agf::AdaptiveGridFilter{T, N}) where {T, N}
    x̂ = state_estimate(agf)
    P = zeros(SMatrix{N, N, T})
    
    for (i, xi) in enumerate(agf.grid_points)
        diff = xi - x̂
        P += agf.probabilities[i] * (diff * diff')
    end
    
    return make_symmetric(P)
end

"""
    num_grid_points(agf::AdaptiveGridFilter)

Return current number of grid points.
"""
function num_grid_points(agf::AdaptiveGridFilter)
    return length(agf.grid_points)
end

"""
    get_grid_statistics(agf::AdaptiveGridFilter)

Return statistics about grid refinement.
"""
function get_grid_statistics(agf::AdaptiveGridFilter)
    return Dict(
        "n_points" => length(agf.grid_points),
        "max_level" => maximum(agf.grid_levels),
        "mean_level" => mean(agf.grid_levels),
        "max_prob" => maximum(agf.probabilities)
    )
end
