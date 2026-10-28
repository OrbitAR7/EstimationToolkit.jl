# News

## v0.9.1

- **GPB1 and GPB2 fixed.** GPB1 never restarted its filters from the merged estimate, and GPB2 never merged per-mode estimates (its (i, j) filters ran independently forever, and the joint probabilities used the mode marginals from two steps back). Both effectively behaved like a static multiple-model bank. They now follow the standard algorithms (Bar-Shalom, Li & Kirubarajan 2001, ch. 11), and new tests check that with one mode, or with identical modes, they reproduce a plain Kalman filter.
- **Mode probabilities computed in log space** for IMM, GPB1, GPB2 and the static multiple-model filter. Before, a large innovation (e.g. a maneuver measured with metre-level noise) made every model's likelihood underflow to zero, and the filters silently kept their prior mode probabilities, which is exactly when they need to switch.
- `ExtendedKalmanFilter` now works inside IMM/GPB filter banks (it was missing `get_predicted_innovation_stats`).
- New example: `examples/orbit_maneuver_imm.jl`, maneuver detection in LEO OD with IMM, GPB2 and two single-model EKFs.
- CI on Julia 1.10 and latest. Minimum Julia is now 1.10. `Manifest.toml` is no longer tracked.
- Fixed a multiple-model test whose data didn't match the model it was meant to favor.
