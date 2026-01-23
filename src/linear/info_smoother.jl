"""
    SquareRootInformationSmoother

Square-Root Information Smoother (SRIS) - numerically stable fixed-interval smoother.

The SRIS combines the numerical advantages of the square-root information filter
with the backward smoothing pass. It operates in information space (inverse
covariance) and uses QR factorization for stability.

# Mathematical Formulation

Information form:
```math
Y = P⁻¹ (information matrix)
ŷ = P⁻¹x̂ (information vector)
```

Square-root representation: Y = R̄'R̄

SRIS backward recursion:
```math
ŷₖ|ₙ = ŷₖ|ₖ + Fₖ'Yₖ₊₁|ₖ⁻¹(ŷₖ₊₁|ₙ - ŷₖ₊₁|ₖ)
Yₖ|ₙ = Yₖ|ₖ + Fₖ'(Yₖ₊₁|ₙ - Yₖ₊₁|ₖ)Fₖ
```

Using square-root form with QR factorization for numerical stability.

# Advantages
- Numerically superior to covariance-form RTS
- Guaranteed positive-definiteness
- Better conditioning for ill-conditioned problems
- Twice the numerical precision effectively

# Reference
- EstimationNotes.pdf Ch. 11-12 (SRIF and Smoothing)
- Bierman (1977) - "Factorization Methods for Discrete Sequential Estimation"

# Fields
- `filtered_R::Vector{Matrix{T}}` - Square-root info matrices (forward)
- `filtered_y::Vector{SVector{N,T}}` - Information vectors (forward)
- `predicted_R::Vector{Matrix{T}}` - Predicted square-roots
- `predicted_y::Vector{SVector{N,T}}` - Predicted info vectors
- `smoothed_R::Vector{Matrix{T}}` - Smoothed square-roots
- `smoothed_y::Vector{SVector{N,T}}` - Smoothed info vectors
"""
mutable struct SquareRootInformationSmoother{T<:Real, N}
    # Forward pass storage
    filtered_R::Vector{Matrix{T}}  # R̄ where Y = R̄'R̄
    filtered_y::Vector{SVector{N, T}}
    predicted_R::Vector{Matrix{T}}
    predicted_y::Vector{SVector{N, T}}
    
    # Backward pass results
    smoothed_R::Vector{Matrix{T}}
    smoothed_y::Vector{SVector{N, T}}
    
    # Dynamics matrices
    F_matrices::Vector{Matrix{T}}
end

"""
    SquareRootInformationSmoother(srif::SquareRootInformationFilter, T_steps::Int)

Initialize SRIS from a SRIF and expected number of time steps.
"""
function SquareRootInformationSmoother(
    srif::SquareRootInformationFilter{T, N},
    T_steps::Int
) where {T, N}
    
    # Pre-allocate storage
    filtered_R = [zeros(T, N, N) for _ in 1:T_steps]
    filtered_y = Vector{SVector{N, T}}(undef, T_steps)
    predicted_R = [zeros(T, N, N) for _ in 1:T_steps]
    predicted_y = Vector{SVector{N, T}}(undef, T_steps)
    smoothed_R = [zeros(T, N, N) for _ in 1:T_steps]
    smoothed_y = Vector{SVector{N, T}}(undef, T_steps)
    F_matrices = [zeros(T, N, N) for _ in 1:T_steps]
    
    return SquareRootInformationSmoother{T, N}(
        filtered_R, filtered_y, predicted_R, predicted_y,
        smoothed_R, smoothed_y, F_matrices
    )
end

"""
    forward_pass!(smoother, srif, dynamics, measurements, controls, dt)

Run forward SRIF pass and store all filtered/predicted estimates.
"""
function forward_pass!(
    smoother::SquareRootInformationSmoother{T, N},
    srif::SquareRootInformationFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    model_meas::AbstractMeasurementModel,
    measurements::Vector,
    controls::Vector = fill(zeros(T, 0), length(measurements)),
    dt::T = T(1.0)
) where {T, N}
    
    K = length(measurements)
    @assert K <= length(smoother.filtered_R) "Not enough pre-allocated storage"
    
    for k in 1:K
        # Dynamics prediction
        predict!(srif, model_dyn, controls[k], dt)
        
        # Store predicted information form before update
        smoother.predicted_R[k] = copy(srif.R̄)
        smoother.predicted_y[k] = srif.ȳ
        
        # Measurement update
        update!(srif, model_meas, measurements[k])
        
        # Store filtered information form after update
        smoother.filtered_R[k] = copy(srif.R̄)
        smoother.filtered_y[k] = srif.ȳ
        
        # Store dynamics Jacobian
        x̂ = srif.R̄ \ srif.ȳ  # Recover state estimate
        smoother.F_matrices[k] = jacobian_x(model_dyn, x̂, controls[k], dt)
        
        # Time update (predict next step if not at end)
        if k < K
            predict!(srif, model_dyn, controls[k], dt)
        end
    end
    
    return smoother
end

"""
    backward_pass!(smoother)

Perform backward smoothing pass in information form.
"""
function backward_pass!(smoother::SquareRootInformationSmoother{T, N}) where {T, N}
    K = length(smoother.filtered_R)
    
    # Initialize: at final time, smoothed = filtered
    smoother.smoothed_R[K] = copy(smoother.filtered_R[K])
    smoother.smoothed_y[K] = smoother.filtered_y[K]
    
    # Backward recursion in information form
    for k in (K-1):-1:1
        F = smoother.F_matrices[k]
        
        # Convert to information matrices
        R̄_filt = smoother.filtered_R[k]
        ŷ_filt = smoother.filtered_y[k]
        Y_filt = R̄_filt' * R̄_filt
        
        R̄_pred_next = smoother.predicted_R[k+1]
        ŷ_pred_next = smoother.predicted_y[k+1]
        Y_pred_next = R̄_pred_next' * R̄_pred_next
        
        R̄_smooth_next = smoother.smoothed_R[k+1]
        ŷ_smooth_next = smoother.smoothed_y[k+1]
        Y_smooth_next = R̄_smooth_next' * R̄_smooth_next
        
        # Information smoother recursion
        # Yₖ|ₙ = Yₖ|ₖ + Fₖ'(Yₖ₊₁|ₙ - Yₖ₊₁|ₖ)Fₖ
        Y_diff = Y_smooth_next - Y_pred_next
        Y_smooth = Y_filt + F' * Y_diff * F
        Y_smooth = make_symmetric(Y_smooth)
        
        # ŷₖ|ₙ = ŷₖ|ₖ + Fₖ'Yₖ₊₁|ₖ⁻¹(ŷₖ₊₁|ₙ - ŷₖ₊₁|ₖ)
        y_diff = ŷ_smooth_next - ŷ_pred_next
        C = cholesky(Symmetric(Y_pred_next))
        ŷ_smooth = ŷ_filt + F' * (C \ y_diff)
        
        # Convert back to square-root form
        C_smooth = cholesky(Symmetric(Y_smooth))
        smoother.smoothed_R[k] = Matrix(C_smooth.L')  # Upper triangular
        smoother.smoothed_y[k] = ŷ_smooth
    end
    
    return smoother
end

"""
    smooth!(smoother, srif, dynamics, measurements, controls, dt)

Complete two-pass smoothing: forward SRIF + backward info smoothing.
"""
function smooth!(
    smoother::SquareRootInformationSmoother{T, N},
    srif::SquareRootInformationFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    model_meas::AbstractMeasurementModel,
    measurements::Vector,
    controls::Vector = fill(zeros(T, 0), length(measurements)),
    dt::T = T(1.0)
) where {T, N}
    
    # Forward pass
    forward_pass!(smoother, srif, model_dyn, model_meas, measurements, controls, dt)
    
    # Backward pass
    backward_pass!(smoother)
    
    return smoother
end

"""
    get_smoothed_trajectory(smoother)

Return smoothed state trajectory.
"""
function get_smoothed_trajectory(smoother::SquareRootInformationSmoother{T, N}) where {T, N}
    K = length(smoother.smoothed_R)
    
    states = Vector{SVector{N, T}}(undef, K)
    covs = Vector{SMatrix{N, N, T}}(undef, K)
    
    for k in 1:K
        # Recover state: x̂ = R̄⁻¹ŷ = Y⁻¹ŷ
        states[k] = smoother.smoothed_R[k] \ smoother.smoothed_y[k]
        
        # Recover covariance: P = Y⁻¹ = (R̄'R̄)⁻¹
        Y = smoother.smoothed_R[k]' * smoother.smoothed_R[k]
        C = cholesky(Symmetric(Y))
        P_inv = inv(C)
        covs[k] = SMatrix{N, N, T}(make_symmetric(P_inv))
    end
    
    return states, covs
end

"""
    get_filtered_trajectory(smoother)

Return filtered (forward-only) trajectory.
"""
function get_filtered_trajectory(smoother::SquareRootInformationSmoother{T, N}) where {T, N}
    K = length(smoother.filtered_R)
    
    states = Vector{SVector{N, T}}(undef, K)
    covs = Vector{SMatrix{N, N, T}}(undef, K)
    
    for k in 1:K
        states[k] = smoother.filtered_R[k] \ smoother.filtered_y[k]
        
        Y = smoother.filtered_R[k]' * smoother.filtered_R[k]
        C = cholesky(Symmetric(Y))
        P_inv = inv(C)
        covs[k] = SMatrix{N, N, T}(make_symmetric(P_inv))
    end
    
    return states, covs
end

"""
    get_smoothed_state(smoother, k)

Get smoothed state at specific time step k.
"""
function get_smoothed_state(smoother::SquareRootInformationSmoother{T, N}, k::Int) where {T, N}
    @assert 1 <= k <= length(smoother.smoothed_R) "Index out of bounds"
    return smoother.smoothed_R[k] \ smoother.smoothed_y[k]
end

"""
    get_smoothed_covariance(smoother, k)

Get smoothed covariance at specific time step k.
"""
function get_smoothed_covariance(smoother::SquareRootInformationSmoother{T, N}, k::Int) where {T, N}
    @assert 1 <= k <= length(smoother.smoothed_R) "Index out of bounds"
    
    Y = smoother.smoothed_R[k]' * smoother.smoothed_R[k]
    C = cholesky(Symmetric(Y))
    P = inv(C)
    
    return SMatrix{N, N, T}(make_symmetric(P))
end
