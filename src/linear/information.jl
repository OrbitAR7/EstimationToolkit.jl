"""
    InformationFilter{T<:Real, N}

Information filter implementation using inverse covariance (information matrix) formulation.

The information filter is mathematically equivalent to the Kalman filter but works with:
- Information matrix: Y = P⁻¹
- Information vector: y = P⁻¹ x = Y x

This formulation is advantageous when:
- Measurements are very informative (R very small)
- Initial uncertainty is very large (P₀ → ∞)
- Multiple independent measurements need to be fused

The filter equations are:
- Prediction: Ȳ⁻¹ = F P F' + Q
- Update: Y = Ȳ + H' R⁻¹ H,  y = ȳ + H' R⁻¹ z

# Fields
- `y::SVector{N, T}`: Information vector (Y * x)
- `Y::SMatrix{N, N, T}`: Information matrix (P⁻¹)
- `ν::Union{Nothing, SVector}`: Innovation from last update (diagnostic)
- `S::Union{Nothing, SMatrix}`: Innovation covariance from last update (diagnostic)

# References
- T. Humphreys, Statistical Estimation Theory course notes (UT Austin ASE 381P.6, 2023), §11.1
- Optimal State Estimation (Simon 2006), Ch. 7

# Notes
The filter internally maintains information form but provides standard state estimates
via state_estimate() = Y⁻¹ y and covariance() = Y⁻¹.
"""
mutable struct InformationFilter{T<:Real, N} <: AbstractFilter
    y::SVector{N, T}           # Information vector
    Y::SMatrix{N, N, T}        # Information matrix
    
    # Store innovation statistics for diagnostics
    ν::Union{Nothing, SVector}
    S::Union{Nothing, SMatrix}
    
    function InformationFilter(x0::SVector{N, T}, P0::SMatrix{N, N, T}) where {N, T<:Real}
        @assert size(P0) == (N, N) "Covariance must be N×N"
        @assert is_positive_definite(P0) "Initial covariance must be SPD"
        
        # Convert to information form
        Y0 = inv(P0)
        Y0 = make_symmetric(Y0)
        y0 = Y0 * x0
        
        new{T, N}(y0, Y0, nothing, nothing)
    end
end

# Convenience constructors
function InformationFilter(x0::AbstractVector{T}, P0::AbstractMatrix{T}) where {T<:Real}
    N = length(x0)
    return InformationFilter(SVector{N, T}(x0), SMatrix{N, N, T}(P0))
end

"""
    predict!(inf::InformationFilter, model::AbstractDynamicsModel, u, dt)

Information filter prediction step.

The prediction in information form requires converting to covariance form:
1. P̄ = Y⁻¹  (recover covariance)
2. x̄ = P̄ y = Y⁻¹ y  (recover state)
3. x̄ = F x̄  (propagate state)
4. P̄ = F P F' + Q  (propagate covariance)
5. Ȳ = P̄⁻¹  (convert back to information)
6. ȳ = Ȳ x̄  (convert state to information)

Note: The prediction step is less efficient than the standard Kalman filter
because it requires matrix inversion. The measurement update is more efficient.
"""
function predict!(inf::InformationFilter{T, N}, model::AbstractDynamicsModel, u, dt) where {T, N}
    # Convert to covariance form
    P = inv(inf.Y)
    P = make_symmetric(P)
    x = P * inf.y
    
    # Get linearized dynamics
    F = jacobian_x(model, x, u, dt)
    Q = process_noise_cov(model, dt)
    
    # Propagate state (nonlinear dynamics)
    x̄ = propagate(model, x, u, dt)
    
    # Propagate covariance
    P̄ = F * P * F' + Q
    P̄ = make_symmetric(P̄)
    
    # Convert back to information form
    Ȳ = inv(P̄)
    Ȳ = make_symmetric(Ȳ)
    ȳ = Ȳ * x̄
    
    inf.y = ȳ
    inf.Y = Ȳ
    
    return inf
end

"""
    update!(inf::InformationFilter, model::AbstractMeasurementModel, z)

Information filter measurement update step.

The measurement update in information form is very efficient:
- i = H' R⁻¹ z  (information contribution)
- I = H' R⁻¹ H  (information matrix contribution)
- Y⁺ = Ȳ + I  (additive update!)
- y⁺ = ȳ + i

This additive form is elegant and numerically stable. Multiple measurements
can be processed sequentially by repeated addition.

# Arguments
- `inf::InformationFilter`: The filter to update
- `model::AbstractMeasurementModel`: Measurement model
- `z::AbstractVector`: Measurement vector

# Note
The measurement update does NOT require matrix inversion, which is the main
advantage of the information form for measurement-rich scenarios.
"""
function update!(inf::InformationFilter{T, N}, model::AbstractMeasurementModel, z) where {T, N}
    # Get measurement model
    H = jacobian_x(model, state_estimate(inf))
    R = measurement_noise_cov(model)
    
    # Compute information contributions
    R_inv = inv(R)
    R_inv = make_symmetric(R_inv)
    
    I_meas = H' * R_inv * H  # Information matrix contribution
    I_meas = make_symmetric(I_meas)
    
    i_meas = H' * R_inv * z  # Information vector contribution
    
    # Store innovation for diagnostics (requires covariance form)
    x̄ = state_estimate(inf)
    ŷ = measure(model, x̄)
    ν = z - ŷ
    
    P̄ = covariance(inf)
    S = H * P̄ * H' + R
    S = make_symmetric(S)
    
    inf.ν = ν
    inf.S = S
    
    # Additive update in information form
    inf.Y = inf.Y + I_meas
    inf.Y = make_symmetric(inf.Y)
    inf.y = inf.y + i_meas
    
    return inf
end

"""
    step!(inf::InformationFilter, dyn, meas, u, z, dt)

Combined prediction and update step.
"""
function step!(inf::InformationFilter, dyn::AbstractDynamicsModel, 
               meas::AbstractMeasurementModel, u, z, dt)
    predict!(inf, dyn, u, dt)
    update!(inf, meas, z)
    return inf
end

"""
    state_estimate(inf::InformationFilter)

Recover state estimate from information form: x̂ = Y⁻¹ y
"""
function state_estimate(inf::InformationFilter{T, N}) where {T, N}
    P = inv(inf.Y)
    return SVector{N, T}(P * inf.y)
end

"""
    covariance(inf::InformationFilter)

Recover covariance from information form: P = Y⁻¹
"""
function covariance(inf::InformationFilter{T, N}) where {T, N}
    P = inv(inf.Y)
    return SMatrix{N, N, T}(make_symmetric(P))
end

"""
    innovation(inf::InformationFilter)

Return the innovation (measurement residual) from the last update.
"""
function innovation(inf::InformationFilter)
    return inf.ν
end

"""
    innovation_covariance(inf::InformationFilter)

Return the innovation covariance from the last update.
"""
function innovation_covariance(inf::InformationFilter)
    return inf.S
end

"""
    normalized_innovation_squared(inf::InformationFilter)

Compute NIS statistic: ν' S⁻¹ ν
"""
function normalized_innovation_squared(inf::InformationFilter)
    if isnothing(inf.ν) || isnothing(inf.S)
        return nothing
    end
    
    return inf.ν' * (inf.S \ inf.ν)
end
