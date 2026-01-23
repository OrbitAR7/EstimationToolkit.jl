"""
    GaussianBelief{T<:Real, N}

Represents a Gaussian belief state with mean and covariance.

# Fields
- `μ::SVector{N, T}`: Mean vector
- `Σ::SMatrix{N, N, T}`: Covariance matrix (must be symmetric positive definite)

# Example
```julia
μ = SVector{2}(1.0, 2.0)
Σ = SMatrix{2,2}(1.0, 0.0, 0.0, 1.0)
belief = GaussianBelief(μ, Σ)
```
"""
struct GaussianBelief{T<:Real, N}
    μ::SVector{N, T}
    Σ::SMatrix{N, N, T}
    
    function GaussianBelief(μ::SVector{N, T}, Σ::SMatrix{N, N, T}) where {N, T<:Real}
        # Basic sanity checks - don't want to be too strict on construction
        # since intermediate steps might temporarily violate SPD
        if !issymmetric(Σ)
            @warn "Covariance matrix is not symmetric, symmetrizing..."
            Σ = SMatrix{N,N,T}(0.5 * (Σ + Σ'))
        end
        new{T, N}(μ, Σ)
    end
end

# Convenience constructor for vectors/matrices
function GaussianBelief(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Real}
    N = length(μ)
    @assert size(Σ) == (N, N) "Dimension mismatch: μ has length $N but Σ is $(size(Σ))"
    return GaussianBelief(SVector{N,T}(μ), SMatrix{N,N,T}(Σ))
end

Base.eltype(::GaussianBelief{T,N}) where {T,N} = T
Base.length(::GaussianBelief{T,N}) where {T,N} = N

"""
    ParticleBelief{T<:Real, N}

Represents a particle-based belief state with weighted samples.

# Fields
- `particles::Vector{SVector{N, T}}`: Particle states
- `weights::Vector{T}`: Normalized weights (should sum to 1)

# Example
```julia
particles = [SVector{2}(randn(), randn()) for _ in 1:100]
weights = fill(1.0/100, 100)
belief = ParticleBelief(particles, weights)
```
"""
struct ParticleBelief{T<:Real, N}
    particles::Vector{SVector{N, T}}
    weights::Vector{T}
    
    function ParticleBelief(particles::Vector{SVector{N, T}}, weights::Vector{T}) where {N, T<:Real}
        @assert length(particles) == length(weights) "Number of particles must match number of weights"
        
        # Normalize weights if they don't sum to 1 (common case)
        w_sum = sum(weights)
        if !isapprox(w_sum, 1.0, atol=1e-10)
            weights = weights ./ w_sum
        end
        
        new{T, N}(particles, weights)
    end
end

Base.eltype(::ParticleBelief{T,N}) where {T,N} = T
Base.length(::ParticleBelief{T,N}) where {T,N} = N
num_particles(p::ParticleBelief) = length(p.particles)

"""
    GMMBelief{T<:Real, N}

Represents a Gaussian mixture model belief state.

# Fields
- `components::Vector{GaussianBelief{T, N}}`: Mixture components
- `weights::Vector{T}`: Component weights (should sum to 1)
"""
struct GMMBelief{T<:Real, N}
    components::Vector{GaussianBelief{T, N}}
    weights::Vector{T}
    
    function GMMBelief(components::Vector{GaussianBelief{T, N}}, weights::Vector{T}) where {N, T<:Real}
        @assert length(components) == length(weights) "Number of components must match number of weights"
        
        # Normalize weights
        w_sum = sum(weights)
        if !isapprox(w_sum, 1.0, atol=1e-10)
            weights = weights ./ w_sum
        end
        
        new{T, N}(components, weights)
    end
end

Base.eltype(::GMMBelief{T,N}) where {T,N} = T
Base.length(::GMMBelief{T,N}) where {T,N} = N
num_components(g::GMMBelief) = length(g.components)
