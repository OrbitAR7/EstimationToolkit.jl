"""
    MarginalizedPointMassFilter{T<:Real, NL, NN}

Marginalized Point Mass Filter - Hybrid grid + Kalman filter.

!!! warning "Experimental"
    This filter is currently experimental and may have API instabilities.
    Use with caution. Requires careful model structure specification.

The marginalized PMF exploits conditionally linear structure by using
a Kalman filter for the linear substate and a grid for the nonlinear substate.
This is the grid-based analog of the Rao-Blackwellized particle filter.

# State Structure

Partition: x = [xₗ, xₙ]
- xₗ: Linear substate (Gaussian given xₙ)
- xₙ: Nonlinear substate (grid-based)

# Mathematical Formulation

For each grid point representing xₙ⁽ⁱ⁾:
- Maintain probability: pᵢ = P(xₙ = xₙ⁽ⁱ⁾)
- Run Kalman filter: xₗ⁽ⁱ⁾ ~ N(x̂ₗ⁽ⁱ⁾, Pₗ⁽ⁱ⁾) | xₙ⁽ⁱ⁾

Overall estimate:
```math
x̂ = [∑ᵢ pᵢ x̂ₗ⁽ⁱ⁾, ∑ᵢ pᵢ xₙ⁽ⁱ⁾]
```

# Advantages
- Optimal estimation of linear substate
- Fewer grid points needed than full PMF
- Handles higher-dimensional problems

# Limitations
- Requires conditionally linear structure
- Still limited by nonlinear substate dimension

# Reference
- 03-Nonlinear_filters.pdf §4.3 (Marginalized PMF)
- Šimandl et al. (2006)

# Fields
- `grid_points_nonlinear::Vector{SVector{NN,T}}` - Grid for nonlinear substate
- `probabilities::Vector{T}` - Grid point probabilities
- `kalman_filters::Vector{KalmanFilter{T,NL}}` - Kalman filters for linear substate
"""
mutable struct MarginalizedPointMassFilter{T<:Real, NL, NN, KF<:KalmanFilter} <: AbstractFilter
    # Nonlinear grid
    grid_points_nonlinear::Vector{SVector{NN, T}}
    probabilities::Vector{T}
    bounds_nonlinear::Vector{Tuple{T, T}}
    
    # Kalman filters for linear substate (one per grid point)
    kalman_filters::Vector{KF}
    
    # Storage
    predicted_probabilities::Vector{T}
end

"""
    MarginalizedPointMassFilter(bounds_nonlinear, n_points_nonlinear, μ0_linear, P0_linear, x0_nonlinear)

Create marginalized PMF.

# Arguments
- `bounds_nonlinear::Vector{Tuple{T,T}}` - Bounds for nonlinear substate
- `n_points_nonlinear::Vector{Int}` - Grid points for nonlinear substate
- `μ0_linear::SVector{NL,T}` - Initial mean for linear substate
- `P0_linear::SMatrix{NL,NL,T}` - Initial covariance for linear substate
- `x0_nonlinear::SVector{NN,T}` - Initial nonlinear substate
"""
function MarginalizedPointMassFilter(
    bounds_nonlinear::Vector{Tuple{T, T}},
    n_points_nonlinear::Vector{Int},
    μ0_linear::SVector{NL, T},
    P0_linear::SMatrix{NL, NL, T},
    x0_nonlinear::SVector{NN, T}
) where {T<:Real, NL, NN}
    
    # Generate grid for nonlinear substate
    grid_points = generate_uniform_grid(bounds_nonlinear, n_points_nonlinear)
    n_grid = length(grid_points)
    
    # Initialize probabilities (concentrated at x0_nonlinear)
    probabilities = zeros(T, n_grid)
    closest_idx = find_closest_grid_point(x0_nonlinear, grid_points)
    probabilities[closest_idx] = T(1.0)
    
    # Initialize Kalman filter for each grid point
    kalman_filters = [KalmanFilter(μ0_linear, P0_linear) for _ in 1:n_grid]
    
    predicted_probabilities = copy(probabilities)
    
    return MarginalizedPointMassFilter{T, NL, NN, KalmanFilter{T, NL}}(
        grid_points, probabilities, bounds_nonlinear,
        kalman_filters, predicted_probabilities
    )
end

"""
    predict!(mpf::MarginalizedPointMassFilter, model_dyn, u, dt)

Prediction step for marginalized PMF.

# Model Requirements
- `propagate_nonlinear(xₙ, xₗ, u, dt)` - Nonlinear dynamics
- `linear_dynamics_matrix(xₙ, u, dt)` - A matrix for linear substate
- `process_noise_cov_nonlinear(xₙ, dt)` - Q for nonlinear state
- `process_noise_cov_linear(xₙ, dt)` - Q for linear state
"""
function predict!(
    mpf::MarginalizedPointMassFilter{T, NL, NN, KF},
    model_dyn,
    u = nothing,
    dt::T = T(1.0)
) where {T, NL, NN, KF}
    
    n_grid = length(mpf.grid_points_nonlinear)
    mpf.predicted_probabilities = zeros(T, n_grid)
    
    # For each current grid point
    for j in 1:n_grid
        if mpf.probabilities[j] < eps(T)
            continue
        end
        
        xₙ_j = mpf.grid_points_nonlinear[j]
        xₗ_mean_j = state_estimate(mpf.kalman_filters[j])
        
        # Propagate nonlinear state
        xₙ_prop = propagate_nonlinear(model_dyn, xₙ_j, xₗ_mean_j, u, dt)
        Qₙ = process_noise_cov_nonlinear(model_dyn, xₙ_j, dt)
        
        # Compute transition probabilities to all grid points
        for i in 1:n_grid
            xₙ_i = mpf.grid_points_nonlinear[i]
            diff = xₙ_i - xₙ_prop
            
            # Gaussian transition probability
            if NN == 1
                trans_prob = exp(-0.5 * diff[1]^2 / Qₙ[1,1]) / sqrt(2π * Qₙ[1,1])
            else
                dist = MvNormal(xₙ_prop, Qₙ)
                trans_prob = pdf(dist, xₙ_i)
            end
            
            mpf.predicted_probabilities[i] += trans_prob * mpf.probabilities[j]
        end
    end
    
    # Normalize
    total = sum(mpf.predicted_probabilities)
    if total > eps(T)
        mpf.predicted_probabilities ./= total
    end
    
    # Propagate each Kalman filter for linear substate
    for i in 1:n_grid
        xₙ_i = mpf.grid_points_nonlinear[i]
        kf = mpf.kalman_filters[i]
        
        # Linear dynamics: xₗ(k+1) = A(xₙ) xₗ(k) + noise
        A = linear_dynamics_matrix(model_dyn, xₙ_i, u, dt)
        Qₗ = process_noise_cov_linear(model_dyn, xₙ_i, dt)
        
        # Manual KF prediction
        x̂ = A * kf.x
        P = A * kf.P * A' + Qₗ
        
        kf.x = x̂
        kf.P = make_symmetric(P)
    end
    
    return mpf
end

"""
    update!(mpf::MarginalizedPointMassFilter, model_meas, y)

Measurement update for marginalized PMF.

# Model Requirements
- `linear_measurement_matrix(xₙ)` - H matrix
- `nonlinear_measurement(xₙ)` - h(xₙ) component
- `measurement_noise_cov(xₙ)` - R matrix
"""
function update!(
    mpf::MarginalizedPointMassFilter{T, NL, NN, KF},
    model_meas,
    y::AbstractVector
) where {T, NL, NN, KF}
    
    n_grid = length(mpf.grid_points_nonlinear)
    likelihoods = zeros(T, n_grid)
    
    # For each grid point
    for i in 1:n_grid
        xₙ_i = mpf.grid_points_nonlinear[i]
        kf = mpf.kalman_filters[i]
        
        # Get measurement model components
        H = linear_measurement_matrix(model_meas, xₙ_i)
        h = nonlinear_measurement(model_meas, xₙ_i)
        R = measurement_noise_cov(model_meas, xₙ_i)
        
        # Predicted measurement
        ŷ = H * kf.x + h
        
        # Innovation
        ν = y - ŷ
        
        # Innovation covariance
        S = H * kf.P * H' + R
        S = make_symmetric(S)
        
        # Marginal likelihood (for grid point probability update)
        log_like = -0.5 * (dot(ν, S \ ν) + logdet(S) + length(y) * log(2π))
        likelihoods[i] = exp(log_like)
        
        # Kalman update for linear substate
        K = kf.P * H' / S
        kf.x = kf.x + K * ν
        kf.P = (I - K * H) * kf.P
        kf.P = make_symmetric(kf.P)
    end
    
    # Update grid point probabilities (Bayes rule)
    mpf.probabilities .= likelihoods .* mpf.predicted_probabilities
    
    total = sum(mpf.probabilities)
    if total > eps(T)
        mpf.probabilities ./= total
    else
        mpf.probabilities .= mpf.predicted_probabilities
    end
    
    return mpf
end

"""
    step!(mpf, model_dyn, model_meas, u, y, dt)

Combined predict and update.
"""
function step!(
    mpf::MarginalizedPointMassFilter{T, NL, NN, KF},
    model_dyn,
    model_meas,
    u::AbstractVector,
    y::AbstractVector,
    dt::T = T(1.0)
) where {T, NL, NN, KF}
    
    predict!(mpf, model_dyn, u, dt)
    update!(mpf, model_meas, y)
    return mpf
end

"""
    state_estimate(mpf::MarginalizedPointMassFilter)

Compute MMSE estimate of full state [xₗ, xₙ].
"""
function state_estimate(mpf::MarginalizedPointMassFilter{T, NL, NN, KF}) where {T, NL, NN, KF}
    # Weighted average of nonlinear state
    x̂ₙ = zeros(SVector{NN, T})
    for (i, xₙ_i) in enumerate(mpf.grid_points_nonlinear)
        x̂ₙ += mpf.probabilities[i] * xₙ_i
    end
    
    # Weighted average of linear state
    x̂ₗ = zeros(SVector{NL, T})
    for i in eachindex(mpf.kalman_filters)
        x̂ₗ += mpf.probabilities[i] * state_estimate(mpf.kalman_filters[i])
    end
    
    return vcat(x̂ₗ, x̂ₙ)
end

"""
    covariance(mpf::MarginalizedPointMassFilter)

Compute covariance of full state.
"""
function covariance(mpf::MarginalizedPointMassFilter{T, NL, NN, KF}) where {T, NL, NN, KF}
    x̂ = state_estimate(mpf)
    x̂ₗ = x̂[1:NL]
    x̂ₙ = x̂[(NL+1):end]
    
    P_full = zeros(SMatrix{NL+NN, NL+NN, T})
    
    for i in eachindex(mpf.grid_points_nonlinear)
        xₗ_i = state_estimate(mpf.kalman_filters[i])
        xₙ_i = mpf.grid_points_nonlinear[i]
        x_i = vcat(xₗ_i, xₙ_i)
        
        diff = x_i - x̂
        P_full += mpf.probabilities[i] * (diff * diff')
        
        # Add linear substate covariance
        Pₗ_i = covariance(mpf.kalman_filters[i])
        P_block = zeros(SMatrix{NL+NN, NL+NN, T})
        for ii in 1:NL
            for jj in 1:NL
                P_block = setindex(P_block, Pₗ_i[ii,jj], ii, jj)
            end
        end
        P_full += mpf.probabilities[i] * P_block
    end
    
    return make_symmetric(P_full)
end

"""
    num_grid_points(mpf::MarginalizedPointMassFilter)

Return number of grid points.
"""
function num_grid_points(mpf::MarginalizedPointMassFilter)
    return length(mpf.grid_points_nonlinear)
end
