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
**Lecture Source**: EstimationNotes Ch. 11, 04-linear_filters.pdf §7

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
1. Propagate state: x̄ = f(x, u, dt)
2. Compute Jacobians: F = ∂f/∂x, G = ∂f/∂v
3. Form augmented matrix: M = [S·F' ; √Q·G']
4. QR factorization: M = [S̄']·R where S̄ is the new square-root factor
"""
function predict!(filter::SquareRootCovarianceFilter{T,N}, model::AbstractDynamicsModel, u, dt) where {T,N}
    # Propagate state mean
    filter.x = propagate(model, filter.x, u, dt)
    
    # Get linearized dynamics
    F = jacobian_x(model, filter.x, u, dt)
    Q = process_noise_cov(model, dt)
    
    # For square-root filter, we need sqrt(Q)
    Q_sym = make_symmetric(Q)
    Q_spd = ensure_spd(Q_sym)
    S_Q = SMatrix{N,N,T}(cholesky(Symmetric(Matrix(Q_spd))).L)
    
    # Form augmented matrix for QR factorization
    # M = [S*F' ; S_Q]'  (transposed for QR)
    # After QR: M' = Q*R, where R' is the new S̄
    M_top = filter.S * F'
    M = vcat(M_top, S_Q)
    
    # QR factorization: M = Q*R
    # We want the upper triangular R such that M'*M = R'*R
    # which means P̄ = M*M' = (Q*R)*(Q*R)' = R*R'
    qr_result = qr(Matrix(M))
    R = qr_result.R
    
    # Extract upper triangular part and transpose to get lower triangular S̄
    # Since P̄ = R'*R, we need S̄ such that P̄ = S̄*S̄'
    # Taking S̄ = R' gives us the Cholesky factor
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
    
    # Covariance update using Potter's square-root algorithm
    # Form the matrix for QR factorization
    # Updated covariance: P = (I - K*H)*P̄*(I - K*H)' + K*R*K' (Joseph form)
    # For square-root: Form [S̄*(I-K*H)' ; K*√R]' and do QR
    
    I_KH = SMatrix{N,N,T}(I) - K_static * H
    R_sym = make_symmetric(R)
    R_spd = ensure_spd(R_sym)
    S_R = SMatrix{NY,NY,T}(cholesky(Symmetric(Matrix(R_spd))).L)
    
    # Form augmented matrix
    M_top = filter.S * I_KH'
    M_bottom = K_static * S_R
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
