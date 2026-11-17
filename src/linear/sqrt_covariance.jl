"""
    SquareRootCovarianceFilter

Square-Root Covariance Kalman Filter using Cholesky factorization.

The square-root covariance filter maintains the Cholesky factor S of the covariance 
matrix P = S*S', ensuring numerical stability and guaranteed positive-definiteness.
This is particularly important for long-duration filters or ill-conditioned systems.

# Mathematical Formulation

The filter propagates the square-root factor S instead of P:
- **Covariance**: P = S·S'
- **Prediction**: S̄ = chol([S·F' ; √Q])
- **Update**: S = chol(S̄ - S̄·H'·(H·S̄·S̄'·H' + R)⁻¹·H·S̄)

# Numerical Advantages
1. **Guaranteed SPD**: Cholesky factorization ensures P is always positive definite
2. **Improved precision**: Operates on square-root quantities with better conditioning
3. **Reduced roundoff error**: Approximately doubles effective precision
4. **No symmetrization needed**: S·S' is automatically symmetric

**Reference**: Kaminski, Bryson, & Schmidt (1971), "Discrete square root filtering"
**Reference**: G. J. Bierman, Factorization Methods for Discrete Sequential Estimation, Academic Press, 1977

# Fields
- `x::SVector{N,T}` - State estimate
- `S::SMatrix{N,N,T}` - Cholesky factor of covariance (P = S*S')
- `ν::Union{Nothing, SVector}` - Innovation (measurement residual)
- `S_y::Union{Nothing, SMatrix}` - Innovation covariance square root

# Constructor
```julia
SquareRootCovarianceFilter(x0::SVector{N,T}, P0::SMatrix{N,N,T})
```
Initializes from mean x0 and covariance P0 (computes S0 = chol(P0)).

# Example
```julia
# Initialize filter
x0 = @SVector [0.0, 0.0, 1.0, 0.0]
P0 = @SMatrix [10.0 0.0 0.0 0.0;
                0.0 10.0 0.0 0.0;
                0.0 0.0 1.0 0.0;
                0.0 0.0 0.0 1.0]
sqrt_kf = SquareRootCovarianceFilter(x0, P0)

# Filtering cycle
for (u, y) in zip(controls, measurements)
    predict!(sqrt_kf, dynamics_model, u, dt)
    update!(sqrt_kf, measurement_model, y)
end

# Extract covariance
P = covariance(sqrt_kf)  # Returns S*S'
```

# Performance Characteristics
- **Time complexity**: O(n³) for Cholesky factorization
- **Space complexity**: O(n²) for storing S
- **Numerical stability**: Superior to standard Kalman filter
- **Use case**: Long-duration missions, ill-conditioned systems, high-precision requirements
"""
mutable struct SquareRootCovarianceFilter{T<:Real, N} <: AbstractFilter
    x::SVector{N, T}
    S::SMatrix{N, N, T}  # Cholesky factor: P = S*S'
    ν::Union{Nothing, SVector{<:Any, T}}
    S_y::Union{Nothing, SMatrix{<:Any, <:Any, T}}
    
    function SquareRootCovarianceFilter(x0::SVector{N,T}, P0::SMatrix{N,N,T}) where {N,T}
        # Ensure P0 is SPD and compute its Cholesky factor
        P0_sym = make_symmetric(P0)
        P0_spd = ensure_spd(P0_sym)
        
        # Cholesky factorization: P0 = S0 * S0'
        # Julia's cholesky returns L such that P = L*L', so S = L
        S0 = SMatrix{N,N,T}(cholesky(Symmetric(Matrix(P0_spd))).L)
        
        new{T,N}(x0, S0, nothing, nothing)
    end
end

"""
    predict!(filter::SquareRootCovarianceFilter, model::AbstractDynamicsModel, u, dt)

Square-root covariance filter prediction step.

# Algorithm
1. Linearize at the current estimate: F = ∂f/∂x; Q = process_noise_cov(model, x, dt)
2. Propagate state: x̄ = f(x, u, dt)
3. Form the stacked matrix M = [S'·F' ; S_Q'], with S_Q·S_Q' = Q, so M'·M = F·P·F' + Q
4. QR factorization M = Q·R; the new square-root factor is S̄ = R'
"""
function predict!(filter::SquareRootCovarianceFilter{T,N}, model::AbstractDynamicsModel, u, dt) where {T,N}
    # Linearize and evaluate the process noise at the estimate before propagation
    F = jacobian_x(model, filter.x, u, dt)
    Q = process_noise_cov(model, filter.x, dt)

    # Propagate state mean
    filter.x = propagate(model, filter.x, u, dt)

    # Square root of Q (psd_sqrt also handles a singular Q, e.g. noise that
    # enters only through the accelerations)
    Q_sym = make_symmetric(Q)
    S_Q = SMatrix{N,N,T}(psd_sqrt(Q_sym))

    # Stack the transposed factors: M = [S'·F' ; S_Q'], so that
    # M'·M = F·S·S'·F' + S_Q·S_Q' = F·P·F' + Q = P̄
    M = vcat(filter.S' * F', S_Q')

    # QR factorization M = Q·R gives M'·M = R'·R, so S̄ = R' is a square-root
    # factor of P̄ (lower triangular; the signs of its columns do not matter)
    R = qr(Matrix(M)).R
    filter.S = SMatrix{N,N,T}(LowerTriangular(R'))
    
    # Reset innovation statistics
    filter.ν = nothing
    filter.S_y = nothing
end

"""
    update!(filter::SquareRootCovarianceFilter, model::AbstractMeasurementModel, y)

Square-root covariance filter measurement update.

# Algorithm (Potter's Square-Root Update)
1. Compute predicted measurement: ŷ = h(x̄)
2. Innovation: ν = y - ŷ
3. Linearize: H = ∂h/∂x
4. Innovation covariance square-root: S_y = chol(H·S̄·S̄'·H' + R)
5. Kalman gain: K = S̄·S̄'·H'·S_y⁻ᵀ·S_y⁻¹
6. State update: x = x̄ + K·ν
7. Covariance square-root update via QR factorization
"""
function update!(filter::SquareRootCovarianceFilter{T,N}, model::AbstractMeasurementModel, y::SVector{NY,T}) where {T,N,NY}
    # Predicted measurement
    ŷ = measure(model, filter.x)
    
    # Innovation
    filter.ν = y - ŷ
    
    # Measurement Jacobian
    H = jacobian_x(model, filter.x)
    R = measurement_noise_cov(model)
    
    # Compute innovation covariance P_y = H*P̄*H' + R = H*S̄*S̄'*H' + R
    P̄ = filter.S * filter.S'
    P_y = H * P̄ * H' + R
    P_y_sym = make_symmetric(P_y)
    P_y_spd = ensure_spd(P_y_sym)
    
    # Innovation covariance square-root
    S_y = SMatrix{NY,NY,T}(cholesky(Symmetric(Matrix(P_y_spd))).L)
    filter.S_y = S_y
    
    # Kalman gain: K = P̄*H'*P_y⁻¹ = S̄*S̄'*H'*(S_y*S_y')⁻¹
    # Efficient computation: K = (S̄*S̄'*H') / (S_y*S_y')
    K = (P̄ * H') / Matrix(P_y_spd)
    K_static = SMatrix{N,NY,T}(K)
    
    # State update
    filter.x = filter.x + K_static * filter.ν
    
    # Covariance update in square-root form of the Joseph update
    #   P = (I - K·H)·P̄·(I - K·H)' + K·R·K'
    # Stack the transposed factors: M = [S̄'·(I - K·H)' ; S_R'·K'] ((N + NY) × N),
    # so that M'·M = P, then take the triangular factor from a QR.

    I_KH = SMatrix{N,N,T}(I) - K_static * H
    R_sym = make_symmetric(R)
    R_spd = ensure_spd(R_sym)
    S_R = SMatrix{NY,NY,T}(cholesky(Symmetric(Matrix(R_spd))).L)

    # Form augmented matrix
    M_top = filter.S' * I_KH'
    M_bottom = S_R' * K_static'
    M = vcat(M_top, M_bottom)
    
    # QR factorization
    qr_result = qr(Matrix(M))
    R_qr = qr_result.R
    
    # Update square-root factor
    filter.S = SMatrix{N,N,T}(LowerTriangular(R_qr'))
end

"""
    state_estimate(filter::SquareRootCovarianceFilter)

Return current state estimate.
"""
function state_estimate(filter::SquareRootCovarianceFilter)
    return filter.x
end

"""
    covariance(filter::SquareRootCovarianceFilter)

Reconstruct covariance matrix from square-root factor: P = S*S'
"""
function covariance(filter::SquareRootCovarianceFilter)
    return make_symmetric(filter.S * filter.S')
end

"""
    innovation(filter::SquareRootCovarianceFilter)

Return the most recent innovation (measurement residual).
"""
function innovation(filter::SquareRootCovarianceFilter)
    return filter.ν
end

"""
    innovation_covariance(filter::SquareRootCovarianceFilter)

Reconstruct innovation covariance from square-root: S_y*S_y'
"""
function innovation_covariance(filter::SquareRootCovarianceFilter)
    if filter.S_y === nothing
        return nothing
    end
    return make_symmetric(filter.S_y * filter.S_y')
end

"""
    nis(filter::SquareRootCovarianceFilter)

Compute Normalized Innovation Squared (NIS) statistic.
"""
function nis(filter::SquareRootCovarianceFilter)
    if filter.ν === nothing || filter.S_y === nothing
        return nothing
    end
    
    # NIS = ν' * S_y⁻ᵀ * S_y⁻¹ * ν
    # Solve using back-substitution
    S_y_inv_nu = filter.S_y \ Vector(filter.ν)
    return dot(S_y_inv_nu, S_y_inv_nu)
end
