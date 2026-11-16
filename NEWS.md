# News

## v0.10.0 (unreleased)

- **New filter: `AdaptiveGaussianMixtureFilter`.** A Gaussian mixture filter that splits components during propagation when the dynamics become too nonlinear over a component's extent, using the entropy criterion of DeMars, Bishop & Jah (2013) evaluated from the sigma points. Splits are moment-preserving (`split_gaussian`), the split direction is the one in which the dynamics are most nonlinear, and after an update near-identical components are merged with Runnalls' (2007) criterion. `mixture_logpdf` evaluates the mixture density for scoring against Monte Carlo samples.
- **New example: `examples/orbit_uncertainty.jl`.** 3-D orbit (two-body + J2) with a two-revolution tracking gap: EKF, UKF and the adaptive mixture scored against Monte Carlo by KL divergence, then by error and NEES over a radar pass (200 trials). Figures by `examples/plot_orbit_uncertainty.jl`.
- **New example: `examples/nrho_navigation.jl`.** EKF, UKF and the adaptive mixture on the 9:2 lunar near-rectilinear halo orbit, with the dynamics and the orbit from orbital-mechanics-jl. Away from perilune the EKF's density is exact and the re-fitting filters are not; at perilune it is the other way round. Figures by `examples/plot_nrho_navigation.jl`.
- **`GaussianSumFilter` could not be updated.** `update!` read fields that `ExtendedKalmanFilter` does not have and threw a `FieldError` on every call; the filter had no tests. Fixed, with tests (one component reproduces the EKF; weights follow Bayes' rule).
- **Weights computed in log space** in `GaussianSumFilter` and in the bootstrap, SIR, auxiliary, regularized, Gaussian, EKF/UKF-proposal, Rao-Blackwellized and marginal particle filters. Before, a measurement far from every particle or component in units of the measurement noise made every likelihood underflow to zero, and the filters silently reset to uniform weights (or kept the old ones), discarding the measurement. This is the same problem v0.9.1 fixed in the multiple-model filters. New helpers: `reweight_log!`, `log_likelihood`.
  - **Breaking:** the bootstrap particle filter now calls `log_likelihood(model, y, x)`. A model that added a method to `likelihood` for non-Gaussian noise should add it to `log_likelihood` instead.
  - The grid-based filters still multiply by `exp(...)` directly.
- **`is_positive_definite` was wrong for singular static matrices.** StaticArrays' Cholesky does not flag a zero pivot, so `[1 0; 0 0]` as an `SMatrix` passed as positive definite (and `ensure_spd` then returned it unchanged). The check is now done on a dense copy.
- **Regularized particle filter** no longer throws when all the weight is on one particle (zero kernel bandwidth); it falls back to plain resampling for that step.
- **Splitting utilities fixed.** `split_component` kept the full covariance on both halves, which inflated the mixture covariance by δδᵀ; it now removes δδᵀ so the pair has the covariance of the original. **Breaking:** `split_factor` must be in (0, 1) and defaults to 0.5 (was 1.0). `should_split` threw an `UndefVarError` on every call (a local variable shadowed `eigvals`).
- The plotting scripts' setup command now also adds StaticArrays to the temporary environment (the examples load it directly).

## v0.9.1

- **GPB1 and GPB2 fixed.** GPB1 never restarted its filters from the merged estimate, and GPB2 never merged per-mode estimates (its (i, j) filters ran independently forever, and the joint probabilities used the mode marginals from two steps back). Both effectively behaved like a static multiple-model bank. They now follow the standard algorithms (Bar-Shalom, Li & Kirubarajan 2001, ch. 11), and new tests check that with one mode, or with identical modes, they reproduce a plain Kalman filter.
- **Mode probabilities computed in log space** for IMM, GPB1, GPB2 and the static multiple-model filter. Before, a large innovation (e.g. a maneuver measured with metre-level noise) made every model's likelihood underflow to zero, and the filters silently kept their prior mode probabilities, which is exactly when they need to switch.
- `ExtendedKalmanFilter` now works inside IMM/GPB filter banks (it was missing `get_predicted_innovation_stats`).
- New example: `examples/orbit_maneuver_imm.jl`, maneuver detection in LEO OD with IMM, GPB2 and two single-model EKFs.
- CI on Julia 1.10 and latest. Minimum Julia is now 1.10. `Manifest.toml` is no longer tracked.
- Fixed a multiple-model test whose data didn't match the model it was meant to favor.
