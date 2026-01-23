"""
Ensemble Gaussian Mixture Filter (EnGMF) - Converts particle representations
to Gaussian mixtures using kernel density estimation.

This is a hybrid approach that combines the flexibility of particle filters
with the computational efficiency of Gaussian mixtures.
"""

"""
    particles_to_gmm(particles, weights; method=:kmeans, n_components=nothing)

Convert particle representation to Gaussian mixture using clustering or KDE.

# Arguments
- `particles`: Vector of particle states
- `weights`: Vector of particle weights

# Keyword Arguments
- `method`: Conversion method
  - `:kde` - Kernel density estimation with bandwidth selection (default)
  - `:kmeans` - K-means clustering
  - `:equal_weight` - Equal-weight splitting
- `n_components`: Number of Gaussian components (auto if nothing)
- `bandwidth`: KDE bandwidth (auto-selected if nothing)

# Returns
- `GMMBelief{T, N}`: Gaussian mixture approximation
"""
function particles_to_gmm(
    particles::Vector{SVector{N, T}},
    weights::Vector{T};
    method::Symbol = :kde,
    n_components::Union{Nothing, Int} = nothing,
    bandwidth::Union{Nothing, T} = nothing
) where {T<:Real, N}
    
    n_particles = length(particles)
    @assert n_particles == length(weights)
    @assert abs(sum(weights) - 1) < 1e-6
    
    if method == :kde
        return particles_to_gmm_kde(particles, weights; bandwidth=bandwidth)
        
    elseif method == :kmeans
        if n_components === nothing
            # Heuristic: sqrt(n_particles) components
            n_components = max(1, round(Int, sqrt(n_particles)))
        end
        return particles_to_gmm_kmeans(particles, weights, n_components)
        
    elseif method == :equal_weight
        if n_components === nothing
            n_components = min(n_particles, 10)  # Default to 10 components
        end
        return particles_to_gmm_equal_weight(particles, weights, n_components)
        
    else
        error("Unknown method: $method")
    end
end

"""
    particles_to_gmm_kde(particles, weights; bandwidth=nothing)

Convert particles to GMM using kernel density estimation.

Each particle becomes a Gaussian component with bandwidth-scaled covariance.
Uses Silverman's rule for bandwidth if not specified.
"""
function particles_to_gmm_kde(
    particles::Vector{SVector{N, T}},
    weights::Vector{T};
    bandwidth::Union{Nothing, T} = nothing
) where {T<:Real, N}
    
    n_particles = length(particles)
    
    # Compute bandwidth if not specified (Silverman's rule)
    if bandwidth === nothing
        # Sample covariance
        μ = sample_mean(particles, weights)
        Σ = sample_covariance(particles, weights, μ)
        
        # Silverman's rule: h = (4/(d+2))^(1/(d+4)) * n^(-1/(d+4)) * σ
        h = (4 / (N + 2))^(1 / (N + 4)) * n_particles^(-1 / (N + 4))
        
        # Scale by average standard deviation
        avg_std = sqrt(tr(Σ) / N)
        bandwidth = T(h * avg_std)
    end
    
    # Each particle becomes a component with scaled covariance
    components = GaussianBelief{T, N}[]
    component_weights = T[]
    
    for (particle, weight) in zip(particles, weights)
        # Isotropic covariance with bandwidth
        Σ = SMatrix{N, N, T}(bandwidth^2 * I(N))
        comp = GaussianBelief(particle, Σ)
        
        push!(components, comp)
        push!(component_weights, weight)
    end
    
    return GMMBelief(components, component_weights)
end

"""
    particles_to_gmm_kmeans(particles, weights, n_components)

Convert particles to GMM using K-means clustering.

Particles are clustered into n_components groups, and each cluster
is represented by a Gaussian with sample mean and covariance.
"""
function particles_to_gmm_kmeans(
    particles::Vector{SVector{N, T}},
    weights::Vector{T},
    n_components::Int
) where {T<:Real, N}
    
    n_particles = length(particles)
    
    if n_components >= n_particles
        # Each particle becomes its own component
        return particles_to_gmm_kde(particles, weights)
    end
    
    # Initialize cluster centers (K-means++)
    centers = kmeans_plusplus_init(particles, weights, n_components)
    
    # K-means iterations
    max_iter = 50
    for iter in 1:max_iter
        # Assignment step: assign each particle to nearest center
        assignments = [argmin([sum(abs2, p - c) for c in centers]) for p in particles]
        
        # Update step: recompute centers as weighted means
        new_centers = similar(centers)
        for k in 1:n_components
            cluster_idx = findall(==(k), assignments)
            if isempty(cluster_idx)
                # Keep old center if cluster is empty
                new_centers[k] = centers[k]
            else
                cluster_particles = particles[cluster_idx]
                cluster_weights = weights[cluster_idx]
                cluster_weights ./= sum(cluster_weights)  # Normalize
                new_centers[k] = sample_mean(cluster_particles, cluster_weights)
            end
        end
        
        # Check convergence
        if maximum(sum(abs2, new_centers[k] - centers[k]) for k in 1:n_components) < 1e-6
            break
        end
        
        centers = new_centers
    end
    
    # Compute Gaussian components from clusters
    components = GaussianBelief{T, N}[]
    component_weights = T[]
    
    assignments = [argmin([sum(abs2, p - c) for c in centers]) for p in particles]
    
    for k in 1:n_components
        cluster_idx = findall(==(k), assignments)
        
        if isempty(cluster_idx)
            continue
        end
        
        cluster_particles = particles[cluster_idx]
        cluster_weights_raw = weights[cluster_idx]
        total_weight = sum(cluster_weights_raw)
        cluster_weights = cluster_weights_raw ./ total_weight
        
        # Cluster mean and covariance
        μ = sample_mean(cluster_particles, cluster_weights)
        Σ = sample_covariance(cluster_particles, cluster_weights, μ)
        
        # Ensure minimum covariance
        Σ = ensure_spd(Σ, 1e-6)
        
        push!(components, GaussianBelief(μ, Σ))
        push!(component_weights, total_weight)
    end
    
    # Normalize weights
    if !isempty(component_weights)
        component_weights ./= sum(component_weights)
    end
    
    return GMMBelief(components, component_weights)
end

"""
    kmeans_plusplus_init(particles, weights, k)

Initialize K-means centers using K-means++ algorithm.

This ensures centers are well-spread initially.
"""
function kmeans_plusplus_init(
    particles::Vector{SVector{N, T}},
    weights::Vector{T},
    k::Int
) where {T<:Real, N}
    
    n = length(particles)
    centers = SVector{N, T}[]
    
    # First center: random particle (weighted)
    cumsum_weights = cumsum(weights)
    r = rand(T)
    idx = findfirst(>=(r), cumsum_weights)
    push!(centers, particles[idx])
    
    # Remaining centers: proportional to distance from nearest center
    for _ in 2:k
        # Compute distance to nearest center for each particle
        distances = [minimum(sum(abs2, p - c) for c in centers) for p in particles]
        
        # Weighted probability proportional to distance
        probs = weights .* distances
        probs ./= sum(probs)
        
        # Sample next center
        cumsum_probs = cumsum(probs)
        r = rand(T)
        idx = findfirst(>=(r), cumsum_probs)
        push!(centers, particles[idx])
    end
    
    return centers
end

"""
    particles_to_gmm_equal_weight(particles, weights, n_components)

Convert particles to GMM by splitting into equal-weight groups.

Simpler but less adaptive than K-means or KDE.
"""
function particles_to_gmm_equal_weight(
    particles::Vector{SVector{N, T}},
    weights::Vector{T},
    n_components::Int
) where {T<:Real, N}
    
    n_particles = length(particles)
    
    # Sort particles by weight (descending)
    sorted_idx = sortperm(weights, rev=true)
    
    # Split into n_components groups
    group_size = div(n_particles, n_components)
    
    components = GaussianBelief{T, N}[]
    component_weights = T[]
    
    for k in 1:n_components
        # Determine group indices
        start_idx = (k-1) * group_size + 1
        end_idx = k < n_components ? k * group_size : n_particles
        group_idx = sorted_idx[start_idx:end_idx]
        
        group_particles = particles[group_idx]
        group_weights_raw = weights[group_idx]
        total_weight = sum(group_weights_raw)
        group_weights = group_weights_raw ./ total_weight
        
        # Group mean and covariance
        μ = sample_mean(group_particles, group_weights)
        Σ = sample_covariance(group_particles, group_weights, μ)
        Σ = ensure_spd(Σ, 1e-6)
        
        push!(components, GaussianBelief(μ, Σ))
        push!(component_weights, total_weight)
    end
    
    # Normalize
    component_weights ./= sum(component_weights)
    
    return GMMBelief(components, component_weights)
end

"""
    EnsembleGMMFilter{T, N}

Ensemble Gaussian Mixture Filter - maintains both particle and GMM representations.

The filter alternates between:
1. Particle propagation (bootstrap or other proposal)
2. Conversion to GMM for efficient likelihood computation
3. GMM-based update
4. Resampling back to particles

This provides a balance between particle filter flexibility and
Gaussian mixture computational efficiency.
"""
mutable struct EnsembleGMMFilter{T<:Real, N} <: AbstractFilter
    # Particle representation
    particles::Vector{SVector{N, T}}
    weights::Vector{T}
    
    # GMM representation (updated after each step)
    gmm::Union{Nothing, GMMBelief{T, N}}
    
    # Conversion parameters
    conversion_method::Symbol
    n_gmm_components::Union{Nothing, Int}
    
    # Resampling
    resample_threshold::T
    resample_method::Symbol
    
    # Diagnostics
    last_innovation::Union{Nothing, Vector{T}}
    last_innovation_cov::Union{Nothing, Matrix{T}}
    
    function EnsembleGMMFilter{T, N}(
        particles::Vector{SVector{N, T}},
        weights::Vector{T};
        conversion_method::Symbol = :kmeans,
        n_gmm_components::Union{Nothing, Int} = nothing,
        resample_threshold::T = T(0.5),
        resample_method::Symbol = :systematic
    ) where {T<:Real, N}
        
        @assert length(particles) == length(weights)
        @assert abs(sum(weights) - 1) < 1e-6
        
        new{T, N}(
            particles,
            weights,
            nothing,
            conversion_method,
            n_gmm_components,
            resample_threshold,
            resample_method,
            nothing,
            nothing
        )
    end
end

# Basic operations delegate to particle filter core
num_particles(f::EnsembleGMMFilter) = length(f.particles)
effective_particles(f::EnsembleGMMFilter) = effective_sample_size(f.weights)

function state_estimate(f::EnsembleGMMFilter{T, N}) where {T, N}
    return sample_mean(f.particles, f.weights)
end

function covariance(f::EnsembleGMMFilter{T, N}) where {T, N}
    μ = state_estimate(f)
    return sample_covariance(f.particles, f.weights, μ)
end

state_covariance(f::EnsembleGMMFilter) = covariance(f)

innovation(f::EnsembleGMMFilter) = f.last_innovation
innovation_covariance(f::EnsembleGMMFilter) = f.last_innovation_cov

function nis(f::EnsembleGMMFilter{T}) where T
    if f.last_innovation === nothing || f.last_innovation_cov === nothing
        return T(NaN)
    end
    ν = f.last_innovation
    S = f.last_innovation_cov
    return ν' * (S \ ν)
end
