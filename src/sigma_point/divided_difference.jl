"""
    DividedDifferenceFilter

Second-order divided difference filter (DD2) of Nørgaard, Poulsen & Ravn (2000).

The DD2 replaces the Taylor-series derivatives of the EKF by central divided
differences (Stirling's interpolation formula) of step h along the columns sᵢ of
the Cholesky factor of P. It uses 2n + 1 function evaluations and has no tuning
parameter other than h; h = √3 is optimal for a Gaussian prior.

# Mathematical Formulation

For a function g and prior N(x̄, P = S·S'), with g₀ = g(x̄) and gᵢ± = g(x̄ ± h·sᵢ):
```
mean            = (h² - n)/h² · g₀ + 1/(2h²) · Σᵢ (gᵢ⁺ + gᵢ⁻)
S₁[:, i]        = (gᵢ⁺ - gᵢ⁻) / (2h)
S₂[:, i]        = √(h² - 1)/(2h²) · (gᵢ⁺ + gᵢ⁻ - 2g₀)
covariance      = S₁·S₁' + S₂·S₂'
cross-cov (x,g) = S·S₁'
```
The prediction applies this with g = dynamics (plus Q), the update with
g = measurement (plus R). For linear g the result is exactly the Kalman filter,
for any state dimension.

**Reference**: Nørgaard, Poulsen & Ravn (2000), "New developments in state estimation
for nonlinear systems," Automatica 36(11), 1627-1638.
**References**: R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 4: Linear Filters (UT Austin ASE 381P.7, 2023), §3; R. Zanetti, "Other Kalman Filters," ASE 381P.7 lecture slides (UT Austin, 2023)

# Fields
- `x::SVector{N,T}` - State estimate
- `P::SMatrix{N,N,T}` - Covariance matrix
- `ν::Union{Nothing, SVector}` - Innovation
- `S::Union{Nothing, SMatrix}` - Innovation covariance
- `h::T` - Divided-difference step (default √3, must be at least 1)

# Constructor
```julia
DividedDifferenceFilter(x0::SVector{N,T}, P0::SMatrix{N,N,T}; h=sqrt(3.0))
```

# Example
```julia
ddf = DividedDifferenceFilter(x0, P0)
for (u, y) in zip(controls, measurements)
    predict!(ddf, dynamics_model, u, dt)
    update!(ddf, measurement_model, y)
end
```

# Notes
Earlier versions used 2n equally weighted points at x̄ ± h·sᵢ, which give a
covariance of (h²/n)·P: correct only for n = 3 with the default h, and too small by
a factor 2 at every step for a 6-state orbit.
"""
mutable struct DividedDifferenceFilter{T<:Real, N} <: AbstractFilter
    x::SVector{N, T}
    P::SMatrix{N, N, T}
    ν::Union{Nothing, SVector{<:Any, T}}
    S::Union{Nothing, SMatrix{<:Any, <:Any, T}}
    h::T  # Step size parameter
    
    function DividedDifferenceFilter(x0::SVector{N,T}, P0::SMatrix{N,N,T}; 
                                    h::T=sqrt(T(3))) where {N,T}
        h >= 1 || throw(ArgumentError("the divided-difference step h must be at least 1"))
        # Ensure P0 is SPD
        P0_sym = make_symmetric(P0)
        P0_spd = ensure_spd(P0_sym)
        
        new{T,N}(x0, P0_spd, nothing, nothing, h)
    end
end


"""
    _dd2_transform(g, x̄, P, h)

Second-order divided-difference transform of `g` for a prior N(x̄, P). Returns the
mean, the factors S₁ and S₂ (covariance S₁·S₁' + S₂·S₂'), and the Cholesky factor S
of P (cross-covariance S·S₁').
"""
function _dd2_transform(g, x̄::SVector{N,T}, P::SMatrix{N,N,T}, h::T) where {N,T}
    Sx = SMatrix{N,N,T}(cholesky(Symmetric(Matrix(ensure_spd(make_symmetric(P))))).L)
    g0 = g(x̄)
    gp = [g(x̄ + h * Sx[:, i]) for i in 1:N]
    gm = [g(x̄ - h * Sx[:, i]) for i in 1:N]
    m = length(g0)
    mean = (h^2 - N) / h^2 * g0 + sum(gp[i] + gm[i] for i in 1:N) / (2h^2)
    S1 = Matrix{T}(undef, m, N)
    S2 = Matrix{T}(undef, m, N)
    c2 = sqrt(h^2 - 1) / (2h^2)
    for i in 1:N
        S1[:, i] = (gp[i] - gm[i]) / (2h)
        S2[:, i] = c2 * (gp[i] + gm[i] - 2g0)
    end
    return mean, S1, S2, Sx
end

"""
    predict!(ddf::DividedDifferenceFilter, model::AbstractDynamicsModel, u, dt)

DD2 prediction: mean and covariance of the propagated state from 2n + 1
propagations, plus the process noise.
"""
function predict!(ddf::DividedDifferenceFilter{T,N}, model::AbstractDynamicsModel, u, dt) where {T,N}
    x̄, S1, S2, _ = _dd2_transform(x -> propagate(model, x, u, dt), ddf.x, ddf.P, ddf.h)
    P̄ = S1 * S1' + S2 * S2' + Matrix(process_noise_cov(model, dt))
    ddf.x = SVector{N,T}(x̄)
    ddf.P = ensure_spd(make_symmetric(SMatrix{N,N,T}(P̄)))
    ddf.ν = nothing
    ddf.S = nothing
    return ddf
end

"""
    update!(ddf::DividedDifferenceFilter, model::AbstractMeasurementModel, y)

DD2 measurement update: predicted measurement, innovation covariance
S₁·S₁' + S₂·S₂' + R and cross-covariance S·S₁' from 2n + 1 measurement evaluations,
then the Kalman update.
"""
function update!(ddf::DividedDifferenceFilter{T,N}, model::AbstractMeasurementModel, y::SVector{NY,T}) where {T,N,NY}
    ŷ, S1, S2, Sx = _dd2_transform(x -> measure(model, x), ddf.x, ddf.P, ddf.h)
    ddf.ν = y - SVector{NY,T}(ŷ)
    Pyy = S1 * S1' + S2 * S2' + Matrix(measurement_noise_cov(model))
    ddf.S = ensure_spd(make_symmetric(SMatrix{NY,NY,T}(Pyy)))
    Pxy = Matrix(Sx) * S1'
    K = SMatrix{N,NY,T}(Pxy / Matrix(ddf.S))
    ddf.x = ddf.x + K * ddf.ν
    ddf.P = ensure_spd(make_symmetric(ddf.P - K * ddf.S * K'))
    return ddf
end

"""
    state_estimate(ddf::DividedDifferenceFilter)

Return current state estimate.
"""
function state_estimate(ddf::DividedDifferenceFilter)
    return ddf.x
end

"""
    covariance(ddf::DividedDifferenceFilter)

Return current covariance estimate.
"""
function covariance(ddf::DividedDifferenceFilter)
    return ddf.P
end

"""
    innovation(ddf::DividedDifferenceFilter)

Return the most recent innovation.
"""
function innovation(ddf::DividedDifferenceFilter)
    return ddf.ν
end

"""
    innovation_covariance(ddf::DividedDifferenceFilter)

Return the most recent innovation covariance.
"""
function innovation_covariance(ddf::DividedDifferenceFilter)
    return ddf.S
end

"""
    nis(ddf::DividedDifferenceFilter)

Compute Normalized Innovation Squared (NIS) statistic.
"""
function nis(ddf::DividedDifferenceFilter)
    if ddf.ν === nothing || ddf.S === nothing
        return nothing
    end
    return dot(ddf.ν, ddf.S \ Vector(ddf.ν))
end

# Alias for Central Difference Kalman Filter (CDKF)
"""
    CentralDifferenceKalmanFilter

Alias for [`DividedDifferenceFilter`](@ref). Both names refer to the same algorithm.
"""
const CentralDifferenceKalmanFilter = DividedDifferenceFilter
