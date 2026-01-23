# Matrix operation utilities with numerical stability considerations

"""
    is_positive_definite(P; tol=1e-10)

Check if matrix P is symmetric positive definite.

# Arguments
- `P`: Matrix to check
- `tol`: Tolerance for minimum eigenvalue

Returns `true` if P is SPD, `false` otherwise.
"""
function is_positive_definite(P::AbstractMatrix; tol=1e-10)
    # First check if it's positive definite using Cholesky
    # This is fast and usually sufficient
    try
        cholesky(Symmetric(P))
        return true
    catch
        return false
    end
    
    # Fallback: check eigenvalues (more expensive)
    # eigvals_all = eigvals(Symmetric(P))
    # return all(λ -> λ > tol, eigvals_all)
end

"""
    make_symmetric(P)

Force matrix P to be symmetric by averaging with its transpose.

Returns: (P + P') / 2
"""
function make_symmetric(P::AbstractMatrix{T}) where {T}
    return (P + P') / 2
end

function make_symmetric(P::SMatrix{N, N, T}) where {N, T}
    return SMatrix{N, N, T}((P + P') / 2)
end

"""
    ensure_spd(P; ϵ=1e-10)

Ensure matrix P is symmetric positive definite.

If P is not SPD, attempts to make it SPD by:
1. Symmetrizing
2. Adding small regularization to diagonal if needed

This is sometimes necessary due to numerical errors in propagation.
"""
function ensure_spd(P::AbstractMatrix{T}; ϵ=1e-10) where {T}
    P_sym = make_symmetric(P)
    
    if is_positive_definite(P_sym; tol=ϵ)
        return P_sym
    end
    
    # Try adding small regularization
    n = size(P, 1)
    P_reg = P_sym + ϵ * I(n)
    
    if is_positive_definite(P_reg; tol=ϵ)
        @warn "Covariance matrix regularized by adding $(ϵ) to diagonal"
        return P_reg
    end
    
    # Last resort: reconstruct from eigendecomposition
    eigen_decomp = eigen(Symmetric(P_sym))
    λ_min = minimum(eigen_decomp.values)
    
    if λ_min < ϵ
        # Make all eigenvalues at least ϵ
        λ_fixed = max.(eigen_decomp.values, ϵ)
        P_fixed = eigen_decomp.vectors * Diagonal(λ_fixed) * eigen_decomp.vectors'
        @warn "Covariance matrix reconstructed from eigendecomposition (min eigenvalue was $λ_min)"
        return P_fixed
    end
    
    return P_sym
end

"""
    joseph_form_update(P̄, K, H, R)

Compute covariance update using Joseph form for numerical stability.

Joseph form: P = (I - KH) P̄ (I - KH)' + K R K'

This is more expensive than P = (I - KH) P̄ but guarantees symmetry and
positive definiteness (assuming P̄ and R are SPD).
"""
function joseph_form_update(P̄, K, H, R)
    I_KH = I - K * H
    return I_KH * P̄ * I_KH' + K * R * K'
end

"""
    sqrt_covariance_update(S, K, H, R)

Update square-root covariance S where P = S * S'.

Uses QR decomposition for numerical stability.
"""
function sqrt_covariance_update(S, K, H, R)
    # Form the matrix [S(I-KH)'; sqrt(R)K']
    I_KH = I - K * H
    SR = cholesky(Symmetric(R)).L
    
    M = [S * I_KH'; SR * K']
    
    # QR decomposition
    _, R_qr = qr(M)
    
    # Extract the upper-left block
    n = size(S, 1)
    S_new = R_qr[1:n, 1:n]'
    
    return S_new
end

"""
    solve_spd(P, b)

Solve P*x = b where P is symmetric positive definite.

Uses Cholesky decomposition for efficiency and stability.
"""
function solve_spd(P::AbstractMatrix, b::AbstractVector)
    C = cholesky(Symmetric(P))
    return C \ b
end

function solve_spd(P::AbstractMatrix, B::AbstractMatrix)
    C = cholesky(Symmetric(P))
    return C \ B
end
