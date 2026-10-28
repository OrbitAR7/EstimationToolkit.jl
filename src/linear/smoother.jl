"""
    RauchTungStriebel

Rauch-Tung-Striebel (RTS) fixed-interval smoother for linear Gaussian systems.

The RTS smoother performs a backward pass through forward-filtered estimates to
produce optimal smoothed state estimates that incorporate information from all
measurements (past, present, and future). This is a two-pass algorithm:
1. Forward pass: Run Kalman filter to get filtered estimates
2. Backward pass: Smooth estimates using future information

The smoothed estimate at time k uses all measurements y₁:ₙ (entire sequence),
whereas the filtered estimate only uses y₁:ₖ.

# Mathematical Formulation

Given forward-filtered estimates x̂ₖ₊₁|ₖ, Pₖ₊₁|ₖ (predicted) and x̂ₖ|ₖ, Pₖ|ₖ (filtered):

Smoother gain:
```math
Cₖ = Pₖ|ₖ Fₖᵀ Pₖ₊₁|ₖ⁻¹
```

Smoothed state:
```math
x̂ₖ|ₙ = x̂ₖ|ₖ + Cₖ(x̂ₖ₊₁|ₙ - x̂ₖ₊₁|ₖ)
```

Smoothed covariance:
```math
Pₖ|ₙ = Pₖ|ₖ + Cₖ(Pₖ₊₁|ₙ - Pₖ₊₁|ₖ)Cₖᵀ
```

# Reference
- T. Humphreys, Statistical Estimation Theory course notes (UT Austin ASE 381P.6, 2023), Ch. 12
- Rauch, Tung, and Striebel (1965)
- Anderson & Moore (1979) - Optimal Filtering

# Fields
- `filtered_states::Vector{SVector{N,T}}` - Forward-filtered state estimates
- `filtered_covs::Vector{SMatrix{N,N,T}}` - Forward-filtered covariances
- `predicted_states::Vector{SVector{N,T}}` - One-step predicted states
- `predicted_covs::Vector{SMatrix{N,N,T}}` - One-step predicted covariances
- `smoothed_states::Vector{SVector{N,T}}` - Smoothed state estimates
- `smoothed_covs::Vector{SMatrix{N,N,T}}` - Smoothed covariances
"""
mutable struct RauchTungStriebel{T<:Real, N}
    # Forward pass storage
    filtered_states::Vector{SVector{N, T}}
    filtered_covs::Vector{SMatrix{N, N, T}}
    predicted_states::Vector{SVector{N, T}}
    predicted_covs::Vector{SMatrix{N, N, T}}
    
    # Backward pass results
    smoothed_states::Vector{SVector{N, T}}
    smoothed_covs::Vector{SMatrix{N, N, T}}
    
    # Store dynamics Jacobians for backward pass
    F_matrices::Vector{SMatrix{N, N, T}}
end

"""
    RauchTungStriebel(kf::KalmanFilter, T_steps::Int)

Initialize RTS smoother from a Kalman filter and expected number of time steps.
"""
function RauchTungStriebel(kf::KalmanFilter{T, N}, T_steps::Int) where {T, N}
    # Pre-allocate storage
    filtered_states = Vector{SVector{N, T}}(undef, T_steps)
    filtered_covs = Vector{SMatrix{N, N, T}}(undef, T_steps)
    predicted_states = Vector{SVector{N, T}}(undef, T_steps)
    predicted_covs = Vector{SMatrix{N, N, T}}(undef, T_steps)
    smoothed_states = Vector{SVector{N, T}}(undef, T_steps)
    smoothed_covs = Vector{SMatrix{N, N, T}}(undef, T_steps)
    F_matrices = Vector{SMatrix{N, N, T}}(undef, T_steps)
    
    return RauchTungStriebel{T, N}(
        filtered_states, filtered_covs, 
        predicted_states, predicted_covs,
        smoothed_states, smoothed_covs,
        F_matrices
    )
end

"""
    forward_pass!(smoother, kf, dynamics, measurements, controls, dt)

Run forward Kalman filter pass and store all filtered/predicted estimates.
"""
function forward_pass!(
    smoother::RauchTungStriebel{T, N},
    kf::KalmanFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    model_meas::AbstractMeasurementModel,
    measurements::Vector,
    controls::Vector = fill(zeros(T, 0), length(measurements)),
    dt::T = T(1.0)
) where {T, N}
    
    K = length(measurements)
    @assert K <= length(smoother.filtered_states) "Not enough pre-allocated storage"
    
    for k in 1:K
        # State prediction
        predict!(kf, model_dyn, controls[k], dt)
        
        # Store predicted state before update
        smoother.predicted_states[k] = kf.x
        smoother.predicted_covs[k] = kf.P
        
        # Measurement update
        update!(kf, model_meas, measurements[k])
        
        # Store filtered state after update
        smoother.filtered_states[k] = kf.x
        smoother.filtered_covs[k] = kf.P
        
        # Store dynamics Jacobian for backward pass
        smoother.F_matrices[k] = jacobian_x(model_dyn, kf.x, controls[k], dt)
    end
    
    return smoother
end

"""
    backward_pass!(smoother)

Perform backward smoothing pass to compute smoothed estimates.

Starting from the last time step (where smoothed = filtered), recursively
compute smoothed estimates backward in time using the RTS recursion.
"""
function backward_pass!(smoother::RauchTungStriebel{T, N}) where {T, N}
    K = length(smoother.filtered_states)
    
    # Initialize: at final time, smoothed = filtered
    smoother.smoothed_states[K] = smoother.filtered_states[K]
    smoother.smoothed_covs[K] = smoother.filtered_covs[K]
    
    # Backward recursion
    for k in (K-1):-1:1
        # Smoother gain: Cₖ = Pₖ|ₖ Fₖᵀ Pₖ₊₁|ₖ⁻¹
        F = smoother.F_matrices[k]
        P_filt = smoother.filtered_covs[k]
        P_pred_next = smoother.predicted_covs[k+1]
        
        C = P_filt * F' / P_pred_next  # Smoother gain
        
        # Smoothed state: x̂ₖ|ₙ = x̂ₖ|ₖ + Cₖ(x̂ₖ₊₁|ₙ - x̂ₖ₊₁|ₖ)
        x_filt = smoother.filtered_states[k]
        x_pred_next = smoother.predicted_states[k+1]
        x_smooth_next = smoother.smoothed_states[k+1]
        
        smoother.smoothed_states[k] = x_filt + C * (x_smooth_next - x_pred_next)
        
        # Smoothed covariance: Pₖ|ₙ = Pₖ|ₖ + Cₖ(Pₖ₊₁|ₙ - Pₖ₊₁|ₖ)Cₖᵀ
        P_smooth_next = smoother.smoothed_covs[k+1]
        P_diff = P_smooth_next - P_pred_next
        
        smoother.smoothed_covs[k] = P_filt + C * P_diff * C'
        
        # Ensure symmetry and positive-definiteness
        smoother.smoothed_covs[k] = make_symmetric(smoother.smoothed_covs[k])
    end
    
    return smoother
end

"""
    smooth!(smoother, kf, dynamics, measurements, controls, dt)

Complete two-pass smoothing: forward filtering + backward smoothing.
"""
function smooth!(
    smoother::RauchTungStriebel{T, N},
    kf::KalmanFilter{T, N},
    model_dyn::AbstractDynamicsModel,
    model_meas::AbstractMeasurementModel,
    measurements::Vector,
    controls::Vector = fill(zeros(T, 0), length(measurements)),
    dt::T = T(1.0)
) where {T, N}
    
    # Forward pass: run Kalman filter
    forward_pass!(smoother, kf, model_dyn, model_meas, measurements, controls, dt)
    
    # Backward pass: compute smoothed estimates
    backward_pass!(smoother)
    
    return smoother
end

"""
    get_smoothed_trajectory(smoother)

Return smoothed state trajectory and covariances.
"""
function get_smoothed_trajectory(smoother::RauchTungStriebel)
    return smoother.smoothed_states, smoother.smoothed_covs
end

"""
    get_filtered_trajectory(smoother)

Return filtered (forward-only) state trajectory and covariances.
"""
function get_filtered_trajectory(smoother::RauchTungStriebel)
    return smoother.filtered_states, smoother.filtered_covs
end
