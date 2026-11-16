"""
    AdaptiveGaussianMixtureFilter{T, N}

Gaussian mixture filter that splits its components during propagation, when the
dynamics have become too nonlinear over the extent of a component for a single
Gaussian to follow.

A Kalman-type filter (EKF, UKF) carries one mean and one covariance. Through
nonlinear dynamics an initially Gaussian uncertainty does not stay Gaussian: in
orbit problems it stretches along the orbit and bends into the familiar "banana".
The filter's covariance then describes an ellipsoid that the true distribution no
longer occupies. A Gaussian sum filter with a fixed set of components has the same
problem one level down, and one that starts from a single Gaussian never becomes
anything else. This filter starts from whatever mixture it is given (usually one
Gaussian) and adds components where and when they are needed.

# Algorithm

Each component is propagated with a sigma-point transform. From the same sigma
points two predicted covariances are available:

- `P_σ`, the sigma-point covariance, which includes the second-order effects, and
- `P_lin = A A'`, the covariance a linearized propagation would give, with `A`
  built from central differences of the sigma points.

For a linear map they are equal. Their difference in differential entropy,

    ΔH = ½ (log det P_σ - log det P_lin),

measures how much nonlinearity the component has seen in this step. `ΔH` is
accumulated per component, and when the running sum exceeds `split_tol` the
component is split in three *before* the step (so the step is redone with the
narrower components) and the sums start again from zero.

**Split direction.** The sigma points come in pairs ±s_i about the mean, where the
s_i are the principal axes of the component's correlation matrix (which makes the
choice independent of the units of the state). The second difference
`b_i = ½ [f(μ + γ s_i) + f(μ - γ s_i)] - f(μ)` is the curvature of the dynamics
along s_i. The component is split along the axis with the largest
`b_i' P_σ⁻¹ b_i`, i.e. the direction in which the dynamics are most nonlinear
relative to the predicted uncertainty.

**Split.** [`split_gaussian`](@ref): three components, same mean and covariance as
the one they replace.

**Measurement update.** Each component gets a sigma-point Kalman update; the
weights are multiplied by the component's measurement likelihood `N(y; ŷ_j, S_j)`,
in log space.

**Reduction.** After the update, components with negligible weight are dropped,
and the closest pairs are merged (moment-preserving) while a merge costs less than
`merge_tol` or there are more than `max_components`. "Closest" is Runnalls' upper
bound on the Kullback-Leibler divergence caused by the merge.

# Relation to AEGIS

The entropy criterion follows DeMars, Bishop & Jah (2013). Their filter (AEGIS)
integrates the linearized entropy rate alongside the continuous-time dynamics;
here the comparison is made per discrete step from the sigma points, which needs
no Jacobian and works with any `propagate` function. The split library and the
choice of direction also differ (see above and [`split_gaussian`](@ref)). Results
should be expected to agree in character, not digit for digit.

# Constructor

    AdaptiveGaussianMixtureFilter(x0, P0; kwargs...)
    AdaptiveGaussianMixtureFilter(gmm::GMMBelief; kwargs...)

# Keyword arguments
- `split_tol = 0.05`: accumulated entropy difference [nats] that triggers a split.
  `Inf` disables splitting, which leaves a sigma-point Kalman filter per component.
- `shrink = 0.7`: standard-deviation ratio of the split components, see
  [`split_gaussian`](@ref)
- `max_components = 50`: hard limit; a component is not split if that would exceed it
- `weight_threshold = 1e-8`: components below this weight are dropped after an update
- `merge_tol = 1e-4`: merge pairs cheaper than this [nats] after an update
- `α = 1.0, β = 0.0, κ = 0.0`: scaled unscented transform parameters. The defaults
  give the third-degree cubature rule (no centre weight), for which
  `P_σ - P_lin` is positive semidefinite, so `ΔH ≥ 0`.

# References
- DeMars, Bishop & Jah (2013), "Entropy-based approach for uncertainty propagation
  of nonlinear dynamical systems," J. Guidance, Control, and Dynamics 36(4).
- Runnalls (2007), "Kullback-Leibler approach to Gaussian mixture reduction,"
  IEEE Trans. Aerospace and Electronic Systems 43(3).
- Alspach & Sorenson (1972), "Nonlinear Bayesian estimation using Gaussian sum
  approximations," IEEE Trans. Automatic Control 17(4).
- R. Zanetti, Advanced Topics in Estimation Theory class notes, Part 3: Nonlinear
  Estimation (UT Austin ASE 381P.7, 2023), §7 (Gaussian sum filter)
"""
mutable struct AdaptiveGaussianMixtureFilter{T<:Real, N, L} <: AbstractFilter
    means::Vector{SVector{N, T}}
    covs::Vector{SMatrix{N, N, T, L}}
    weights::Vector{T}
    entropy_dev::Vector{T}          # accumulated ΔH per component since its last split

    split_tol::T
    shrink::T
    max_components::Int
    weight_threshold::T
    merge_tol::T

    # sigma-point rule
    α::T
    β::T
    κ::T

    # diagnostics
    n_splits::Int                   # total number of splits so far
    n_merges::Int                   # total number of merges so far
    ν::Union{Nothing, Vector{T}}    # mixture innovation from the last update
    S::Union{Nothing, Matrix{T}}    # mixture innovation covariance from the last update

    function AdaptiveGaussianMixtureFilter(
        means::Vector{SVector{N, T}},
        covs::Vector{SMatrix{N, N, T, L}},
        weights::Vector{T};
        split_tol::Real = 0.05,
        shrink::Real = 0.7,
        max_components::Int = 50,
        weight_threshold::Real = 1e-8,
        merge_tol::Real = 1e-4,
        α::Real = 1.0, β::Real = 0.0, κ::Real = 0.0
    ) where {T<:Real, N, L}
        M = length(means)
        @assert M > 0 "Must have at least one component"
        @assert length(covs) == M && length(weights) == M "means, covs and weights must have the same length"
        @assert all(w -> w >= 0, weights) && abs(sum(weights) - 1) < 1e-6 "Weights must be non-negative and sum to 1"
        @assert all(is_positive_definite, covs) "Component covariances must be SPD"
        @assert split_tol > 0 "split_tol must be positive (use Inf to disable splitting)"
        @assert 0 < shrink < 1 "shrink must be in (0, 1)"
        @assert max_components >= M "max_components is smaller than the initial number of components"
        @assert α > 0 && N + α^2 * (N + κ) - N > 0 "invalid sigma-point parameters"

        new{T, N, L}(copy(means), copy(covs), weights ./ sum(weights), zeros(T, M),
                     T(split_tol), T(shrink), max_components, T(weight_threshold), T(merge_tol),
                     T(α), T(β), T(κ), 0, 0, nothing, nothing)
    end
end

function AdaptiveGaussianMixtureFilter(x0::SVector{N, T}, P0::SMatrix{N, N, T, L};
                                       kwargs...) where {T<:Real, N, L}
    return AdaptiveGaussianMixtureFilter([x0], [P0], [one(T)]; kwargs...)
end

function AdaptiveGaussianMixtureFilter(x0::AbstractVector{T}, P0::AbstractMatrix{T};
                                       kwargs...) where {T<:Real}
    N = length(x0)
    return AdaptiveGaussianMixtureFilter(SVector{N, T}(x0), SMatrix{N, N, T, N * N}(P0); kwargs...)
end

function AdaptiveGaussianMixtureFilter(gmm::GMMBelief{T, N}; kwargs...) where {T<:Real, N}
    means = [c.μ for c in gmm.components]
    covs = [SMatrix{N, N, T, N * N}(c.Σ) for c in gmm.components]
    return AdaptiveGaussianMixtureFilter(means, covs, copy(gmm.weights); kwargs...)
end

# ============================================================================
# Sigma-point machinery
# ============================================================================

"""
    principal_sqrt(P)

Square root `S` of a covariance (`S * S' == P`) whose columns are the principal
axes of the *correlation* matrix, scaled back to state units: with
`D = diag(σ₁, …, σₙ)` and `D⁻¹ P D⁻¹ = U Λ U'`, `S = D U Λ^{1/2}`.

A plain eigendecomposition of P depends on the units of the state (with position
in km and velocity in km/s the "largest" directions are just the position ones).
Working on the correlation matrix removes that dependence.
"""
function principal_sqrt(P::SMatrix{N, N, T}) where {N, T<:Real}
    σ = sqrt.(max.(diag(P), floatmin(T)))
    C = Matrix(P) ./ (σ * σ')
    E = eigen(Symmetric(C))
    λ = max.(E.values, zero(T))
    S = (σ .* E.vectors) .* sqrt.(λ)'
    return SMatrix{N, N, T, N * N}(S)
end

# λ, γ and the weights of the scaled unscented transform for an N-dimensional state
function _sigma_rule(f::AdaptiveGaussianMixtureFilter{T, N}) where {T, N}
    λ = f.α^2 * (N + f.κ) - N
    γ = sqrt(N + λ)
    w0m = λ / (N + λ)
    w0c = w0m + (1 - f.α^2 + f.β)
    wi = 1 / (2 * (N + λ))
    return γ, w0m, w0c, wi
end

"""
    _sigma_transform(g, μ, P, rule)

Push N(μ, P) through `g` with symmetric sigma points along the principal axes.

Returns a named tuple with
- `mean`, `cov`: sigma-point mean and covariance of g(x) (no noise added)
- `A`: central-difference linearization of g in sigma coordinates, so that the
  linearized covariance is `A * A'` and the cross-covariance is `S * A'`
- `S`: the square root of P that was used (columns are the sigma directions)
- `b`: second differences, `b[i] = ½ [g(μ + γ sᵢ) + g(μ - γ sᵢ)] - g(μ)`
"""
function _sigma_transform(g, μ::SVector{N, T}, P::SMatrix{N, N, T}, rule) where {N, T}
    γ, w0m, w0c, wi = rule
    S = principal_sqrt(P)

    y0 = g(μ)
    yp = [g(μ + γ * S[:, i]) for i in 1:N]
    ym = [g(μ - γ * S[:, i]) for i in 1:N]

    ȳ = w0m * y0
    for i in 1:N
        ȳ += wi * (yp[i] + ym[i])
    end

    d0 = y0 - ȳ
    Pyy = w0c * (d0 * d0')
    for i in 1:N
        dp = yp[i] - ȳ
        dm = ym[i] - ȳ
        Pyy += wi * (dp * dp' + dm * dm')
    end

    A = reduce(hcat, [(yp[i] - ym[i]) / (2γ) for i in 1:N])
    b = [(yp[i] + ym[i]) / 2 - y0 for i in 1:N]

    return (mean = ȳ, cov = (Pyy + Pyy') / 2, A = A, S = S, b = b)
end

# ½ (log det P_σ - log det A A'): entropy of the sigma-point prediction minus the
# entropy of the linearized prediction. Zero for a linear map.
function _entropy_difference(r)
    Pσ = Symmetric(Matrix(r.cov))
    isposdef(Pσ) || return 0.0
    ld_lin = logabsdet(Matrix(r.A))[1]          # log |det A|, so log det(A A') = 2 ld_lin
    isfinite(ld_lin) || return 0.0
    return 0.5 * logdet(cholesky(Pσ)) - ld_lin
end

# Index of the sigma direction along which g is most nonlinear, measured by the
# second difference in units of the predicted covariance.
function _most_nonlinear_direction(r)
    Pσ = Symmetric(Matrix(r.cov))
    C = cholesky(Pσ; check = false)
    issuccess(C) || return 1
    best, η_best = 1, -Inf
    for i in eachindex(r.b)
        bi = Vector(r.b[i])
        η = dot(bi, C \ bi)
        if η > η_best
            best, η_best = i, η
        end
    end
    return best
end

# ============================================================================
# Filter operations
# ============================================================================

"""
    predict!(f::AdaptiveGaussianMixtureFilter, model::AbstractDynamicsModel, u, dt)

Propagate every component, splitting those whose accumulated entropy difference
exceeds `split_tol` (see the type docstring). The children of a split are
propagated over the same step and split again if they need it, so one call
refines the mixture as far as `split_tol` and `max_components` allow. Weights
change only through splits.
"""
function predict!(f::AdaptiveGaussianMixtureFilter{T, N, L}, model::AbstractDynamicsModel,
                  u, dt) where {T, N, L}
    Q = SMatrix{N, N, T, L}(process_noise_cov(model, dt))
    g = x -> propagate(model, x, u, dt)
    rule = _sigma_rule(f)

    # Propagate every component and record its entropy difference.
    pending = map(eachindex(f.weights)) do j
        r = _sigma_transform(g, f.means[j], f.covs[j], rule)
        (μ = f.means[j], P = f.covs[j], w = f.weights[j], r = r,
         dev = f.entropy_dev[j] + T(_entropy_difference(r)))
    end
    
    # While there is room, split the component that is furthest over the tolerance
    # (as it was before this step) and redo the step for its three children. The
    # children are candidates in turn, so a step that is strongly nonlinear is
    # refined until every component is within tolerance or the limit is reached.
    # Taking the worst one first spends the component budget where it is needed.
    while length(pending) + 2 <= f.max_components
        worst = argmax([abs(c.dev) for c in pending])
        c = pending[worst]
        abs(c.dev) > f.split_tol || break
        
        s = c.r.S[:, _most_nonlinear_direction(c.r)]
        child_means, P_child, child_weights = split_gaussian(c.μ, c.P, s; shrink = f.shrink)
        Pc = SMatrix{N, N, T, L}(P_child)
        children = map(1:3) do k
            r = _sigma_transform(g, child_means[k], Pc, rule)
            (μ = child_means[k], P = Pc, w = c.w * child_weights[k], r = r,
             dev = T(_entropy_difference(r)))
        end
        splice!(pending, worst, children)
        f.n_splits += 1
    end
    
    means = [SVector{N, T}(c.r.mean) for c in pending]
    covs = [SMatrix{N, N, T, L}(c.r.cov) + Q for c in pending]
    weights = [c.w for c in pending]
    devs = [c.dev for c in pending]
    
    f.means, f.covs, f.weights, f.entropy_dev = means, covs, weights, devs
    return f
end

"""
    update!(f::AdaptiveGaussianMixtureFilter, model::AbstractMeasurementModel, y)

Sigma-point Kalman update of every component, Bayes update of the weights (in log
space), then pruning and merging.
"""
function update!(f::AdaptiveGaussianMixtureFilter{T, N, L}, model::AbstractMeasurementModel,
                 y) where {T, N, L}
    R = Matrix{T}(measurement_noise_cov(model))
    h = x -> measure(model, x)
    rule = _sigma_rule(f)
    yv = Vector{T}(y)
    ny = length(yv)
    M = length(f.weights)

    loglik = Vector{T}(undef, M)
    ŷs = Vector{Vector{T}}(undef, M)
    Ss = Vector{Matrix{T}}(undef, M)

    for j in 1:M
        μ, P = f.means[j], f.covs[j]
        r = _sigma_transform(h, μ, P, rule)

        ŷ = Vector{T}(r.mean)
        Sj = Matrix{T}(r.cov) + R
        Sj = (Sj + Sj') / 2
        Pxy = Matrix(r.S) * Matrix(r.A)'           # cross-covariance, N × ny

        C = cholesky(Symmetric(Sj))
        ν = yv - ŷ
        K = Pxy / C

        f.means[j] = μ + SVector{N, T}(K * ν)
        P_new = Matrix(P) - K * Sj * K'
        f.covs[j] = SMatrix{N, N, T, L}(ensure_spd((P_new + P_new') / 2))

        loglik[j] = -0.5 * (dot(ν, C \ ν) + ny * log(2π) + logdet(C))
        ŷs[j], Ss[j] = ŷ, Sj
    end

    # Innovation statistics of the mixture, from the prior weights: the predicted
    # measurement is itself a Gaussian mixture with these moments.
    ŷ_mix = sum(f.weights[j] * ŷs[j] for j in 1:M)
    S_mix = sum(f.weights[j] * (Ss[j] + (ŷs[j] - ŷ_mix) * (ŷs[j] - ŷ_mix)') for j in 1:M)
    f.ν = yv - ŷ_mix
    f.S = S_mix

    if !reweight_log!(f.weights, loglik)
        @warn "No component has a finite likelihood, keeping old weights"
    end

    # The update re-approximates each component as Gaussian, so the entropy
    # bookkeeping starts again.
    fill!(f.entropy_dev, zero(T))

    prune_components!(f)
    reduce_components!(f)
    return f
end

# ============================================================================
# Mixture management
# ============================================================================

"""
    split!(f::AdaptiveGaussianMixtureFilter, j, s)

Split component `j` along the one-sigma direction `s` (see [`split_gaussian`](@ref)),
regardless of the entropy criterion. The mean and covariance of the mixture are
unchanged.
"""
function split!(f::AdaptiveGaussianMixtureFilter{T, N, L}, j::Int, s::SVector{N, T}) where {T, N, L}
    child_means, P_child, child_weights = split_gaussian(f.means[j], f.covs[j], s; shrink = f.shrink)
    w = f.weights[j]

    f.means[j] = child_means[2]
    f.covs[j] = SMatrix{N, N, T, L}(P_child)
    f.weights[j] = w * child_weights[2]
    f.entropy_dev[j] = zero(T)
    for k in (1, 3)
        push!(f.means, child_means[k])
        push!(f.covs, SMatrix{N, N, T, L}(P_child))
        push!(f.weights, w * child_weights[k])
        push!(f.entropy_dev, zero(T))
    end
    f.n_splits += 1
    return f
end

"""
    prune_components!(f::AdaptiveGaussianMixtureFilter)

Drop components whose weight is below `weight_threshold` (always keeping at least
the heaviest one) and renormalize.
"""
function prune_components!(f::AdaptiveGaussianMixtureFilter)
    keep = findall(w -> w >= f.weight_threshold, f.weights)
    isempty(keep) && (keep = [argmax(f.weights)])
    if length(keep) < length(f.weights)
        f.means = f.means[keep]
        f.covs = f.covs[keep]
        f.weights = f.weights[keep]
        f.entropy_dev = f.entropy_dev[keep]
        f.weights ./= sum(f.weights)
    end
    return f
end

# Moment-preserving merge of two weighted Gaussians
function _merge_pair(w1, μ1, P1, w2, μ2, P2)
    w = w1 + w2
    μ = (w1 * μ1 + w2 * μ2) / w
    d1, d2 = μ1 - μ, μ2 - μ
    P = (w1 * (P1 + d1 * d1') + w2 * (P2 + d2 * d2')) / w
    return w, μ, (P + P') / 2
end

_logdet_spd(P) = logdet(cholesky(Symmetric(Matrix(P))))

"""
    merge_cost(w1, μ1, P1, w2, μ2, P2)

Runnalls' (2007) upper bound on the increase in Kullback-Leibler divergence when
two components of a mixture are replaced by their moment-matched merge:

    B = ½ [(w1 + w2) log det P₁₂ - w1 log det P₁ - w2 log det P₂]

It is zero for identical components and small for light or similar ones.
"""
function merge_cost(w1, μ1, P1, w2, μ2, P2)
    w, _, P = _merge_pair(w1, μ1, P1, w2, μ2, P2)
    return 0.5 * (w * _logdet_spd(P) - w1 * _logdet_spd(P1) - w2 * _logdet_spd(P2))
end

"""
    reduce_components!(f::AdaptiveGaussianMixtureFilter)

Merge the cheapest pair (by [`merge_cost`](@ref)) repeatedly while that cost is
below `merge_tol` or the mixture has more than `max_components`. Merging keeps the
mean and covariance of the mixture.
"""
function reduce_components!(f::AdaptiveGaussianMixtureFilter{T, N, L}) where {T, N, L}
    while length(f.weights) > 1
        M = length(f.weights)
        ld = [_logdet_spd(P) for P in f.covs]
        best, bi, bj = T(Inf), 0, 0
        for i in 1:M-1, j in i+1:M
            w, _, P = _merge_pair(f.weights[i], f.means[i], f.covs[i],
                                  f.weights[j], f.means[j], f.covs[j])
            c = 0.5 * (w * _logdet_spd(P) - f.weights[i] * ld[i] - f.weights[j] * ld[j])
            if c < best
                best, bi, bj = c, i, j
            end
        end

        (best < f.merge_tol || M > f.max_components) || break

        w, μ, P = _merge_pair(f.weights[bi], f.means[bi], f.covs[bi],
                              f.weights[bj], f.means[bj], f.covs[bj])
        f.weights[bi] = w
        f.means[bi] = SVector{N, T}(μ)
        f.covs[bi] = SMatrix{N, N, T, L}(P)
        f.entropy_dev[bi] = max(f.entropy_dev[bi], f.entropy_dev[bj])
        deleteat!(f.weights, bj)
        deleteat!(f.means, bj)
        deleteat!(f.covs, bj)
        deleteat!(f.entropy_dev, bj)
        f.n_merges += 1
    end
    return f
end

# ============================================================================
# Estimates and diagnostics
# ============================================================================

"""
    state_estimate(f::AdaptiveGaussianMixtureFilter)

Mean of the mixture, `Σ wⱼ μⱼ`.
"""
state_estimate(f::AdaptiveGaussianMixtureFilter) = sum(w * μ for (w, μ) in zip(f.weights, f.means))

"""
    covariance(f::AdaptiveGaussianMixtureFilter)

Covariance of the mixture, `Σ wⱼ [Pⱼ + (μⱼ - x̂)(μⱼ - x̂)ᵀ]`. For a strongly
non-Gaussian mixture this is a summary, not a description: use
[`mixture_logpdf`](@ref) or the components themselves.
"""
function covariance(f::AdaptiveGaussianMixtureFilter)
    x̂ = state_estimate(f)
    return sum(w * (P + (μ - x̂) * (μ - x̂)') for (w, μ, P) in zip(f.weights, f.means, f.covs))
end

num_components(f::AdaptiveGaussianMixtureFilter) = length(f.weights)

function effective_components(f::AdaptiveGaussianMixtureFilter{T}) where {T}
    return exp(-sum(w * log(max(w, eps(T))) for w in f.weights))
end

innovation(f::AdaptiveGaussianMixtureFilter) = f.ν
innovation_covariance(f::AdaptiveGaussianMixtureFilter) = f.S

GMMBelief(f::AdaptiveGaussianMixtureFilter) =
    GMMBelief([GaussianBelief(μ, P) for (μ, P) in zip(f.means, f.covs)], copy(f.weights))

"""
    gaussian_logpdf(x, μ, P)

Log density of N(μ, P) at x.
"""
function gaussian_logpdf(x, μ, P)
    C = cholesky(Symmetric(Matrix(P)))
    d = Vector(x - μ)
    return -0.5 * (dot(d, C \ d) + length(d) * log(2π) + logdet(C))
end

"""
    mixture_logpdf(f::AdaptiveGaussianMixtureFilter, x)
    mixture_logpdf(gmm::GMMBelief, x)

Log of the mixture density at `x`, `log Σ wⱼ N(x; μⱼ, Pⱼ)`, evaluated with the
log-sum-exp trick. Averaged over samples of the true distribution this is the
(negative) cross-entropy, a direct score of how well the mixture describes it.
"""
function mixture_logpdf(f::AdaptiveGaussianMixtureFilter, x::AbstractVector{<:Real})
    return _mixture_logpdf(f.weights, f.means, f.covs, x)
end

function mixture_logpdf(gmm::GMMBelief, x::AbstractVector{<:Real})
    return _mixture_logpdf(gmm.weights, [c.μ for c in gmm.components],
                           [c.Σ for c in gmm.components], x)
end

function _mixture_logpdf(weights, means, covs, x)
    terms = [log(w) + gaussian_logpdf(x, μ, P) for (w, μ, P) in zip(weights, means, covs)]
    m = maximum(terms)
    isfinite(m) || return m
    return m + log(sum(exp(t - m) for t in terms))
end

"""
    mixture_logpdf(f::AdaptiveGaussianMixtureFilter, xs::AbstractVector{<:AbstractVector})

Log density at each point of `xs`. The components are factorized once, so this is
much cheaper than calling the single-point method in a loop when scoring a mixture
against a large Monte Carlo sample.
"""
function mixture_logpdf(f::AdaptiveGaussianMixtureFilter{T, N}, xs::AbstractVector{<:AbstractVector}) where {T, N}
    M = length(f.weights)
    # Whitening matrix and log normalization of each component:
    # (x - μ)' P⁻¹ (x - μ) = |L⁻¹ (x - μ)|² with P = L L'
    Linv = Vector{SMatrix{N, N, T, N * N}}(undef, M)
    logc = Vector{T}(undef, M)
    for j in 1:M
        C = cholesky(Symmetric(Matrix(f.covs[j])))
        Linv[j] = SMatrix{N, N, T, N * N}(inv(C.L))
        logc[j] = log(f.weights[j]) - 0.5 * (N * log(2π) + logdet(C))
    end
    
    out = Vector{T}(undef, length(xs))
    terms = Vector{T}(undef, M)
    for (i, x) in enumerate(xs)
        xs_i = SVector{N, T}(x)
        for j in 1:M
            z = Linv[j] * (xs_i - f.means[j])
            terms[j] = logc[j] - 0.5 * dot(z, z)
        end
        m = maximum(terms)
        out[i] = isfinite(m) ? m + log(sum(t -> exp(t - m), terms)) : m
    end
    return out
end
