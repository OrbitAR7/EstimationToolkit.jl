module EstimationToolkit

using LinearAlgebra
using StaticArrays
using ForwardDiff
using Distributions
using Random
using Statistics
using SpecialFunctions

# Core abstractions
include("core/types.jl")
include("core/interfaces.jl")
include("core/diagnostics.jl")

# Models
include("models/dynamics.jl")
include("models/measurements.jl")
include("models/noise.jl")
include("models/linearization.jl")

# Utilities
include("utils/matrix_ops.jl")
include("utils/statistics.jl")

# Linear filters
include("linear/kalman.jl")
include("linear/information.jl")
include("linear/srif.jl")
include("linear/sqrt_covariance.jl")
include("linear/fading_memory.jl")
include("linear/smoother.jl")
include("linear/info_smoother.jl")

# Extended Kalman family
include("extended/ekf.jl")
include("extended/iekf.jl")
include("extended/second_order.jl")

# Sigma-point filters
include("sigma_point/transforms.jl")
include("sigma_point/unscented.jl")
include("sigma_point/cubature.jl")
include("sigma_point/quadrature.jl")
include("sigma_point/divided_difference.jl")

# Particle filters
include("particle/resampling.jl")
include("particle/bootstrap.jl")
include("particle/sir.jl")
include("particle/auxiliary.jl")
include("particle/regularized.jl")
include("particle/gaussian_pf.jl")
include("particle/optimal_proposal.jl")
include("particle/rao_blackwell.jl")
include("particle/marginal_pf.jl")

# Grid-based filters
include("grid/point_mass.jl")
include("grid/adaptive_grid.jl")
include("grid/marginalized_pmf.jl")

# Gaussian mixture filters
include("gaussian_mixture/gsf.jl")
include("gaussian_mixture/splitting.jl")
include("gaussian_mixture/merging.jl")
include("gaussian_mixture/ensemble_gmm.jl")

# Ensemble methods
include("ensemble/enkf.jl")
include("ensemble/etkf.jl")
include("ensemble/ensemble_sqrt.jl")
include("ensemble/localization.jl")

# Multiple model filters
include("multiple_model/static_mm.jl")
include("multiple_model/imm.jl")
include("multiple_model/gpb.jl")

# Particle flow filters
include("particle_flow/gromov.jl")

# Export core types
export GaussianBelief, ParticleBelief, GMMBelief
export AbstractFilter, AbstractDynamicsModel, AbstractMeasurementModel

# Export filters
export KalmanFilter, InformationFilter, SquareRootInformationFilter, SquareRootCovarianceFilter, FadingMemoryFilter
export RauchTungStriebel, SquareRootInformationSmoother
export ExtendedKalmanFilter, IteratedExtendedKalmanFilter, SecondOrderExtendedKalmanFilter
export UnscentedKalmanFilter, CubatureKalmanFilter, QuadratureKalmanFilter, DividedDifferenceFilter, CentralDifferenceKalmanFilter
export BootstrapParticleFilter, SequentialImportanceResampling, AuxiliaryParticleFilter, RegularizedParticleFilter
export GaussianParticleFilter, ExtendedKalmanParticleFilter, UnscentedParticleFilter
export RaoBlackwellizedParticleFilter, MarginalParticleFilter
export PointMassFilter, AdaptiveGridFilter, MarginalizedPointMassFilter
export GaussianSumFilter, EnsembleGMMFilter
export EnsembleKalmanFilter, EnsembleTransformKalmanFilter, EnsembleSquareRootFilter
export StaticMultipleModelFilter, InteractingMultipleModel
export GPB1Filter, GPB2Filter
export GromovFlowFilter, DaumHuangFlowFilter

# Export operations
export predict!, update!, step!
export state_estimate, covariance, state_covariance
export innovation, innovation_covariance
export nis, nees
export set_forgetting_factor!

# Export model operations
export propagate, measure, jacobian_x, jacobian_v, jacobian_w
export process_noise_cov, measurement_noise_cov
export sample_process_noise, sample_measurement_noise
export simulate_trajectory, simulate_measurements

# Export diagnostics
export normalized_innovation_squared, normalized_estimation_error_squared
export consistency_test

# Export utilities
export is_positive_definite, make_symmetric, ensure_spd
export effective_sample_size, sample_mean, sample_covariance
export systematic_resample, stratified_resample, multinomial_resample
export num_particles, effective_particles
export bootstrap_proposal, information_matrix
export ekf_proposal, ukf_proposal

# Export GMM utilities
export split_component, merge_components, prune_components!
export num_components, effective_components
export particles_to_gmm

# Export ensemble utilities
export num_ensemble, ensemble_spread
export gaspari_cohn, gaussian_localization, boxcar_localization
export localization_matrix, apply_localization
export get_ensemble

# Export transforms
export unscented_transform, cubature_transform, transform_sigma_points, reconstruct_gaussian

# Export smoother utilities
export forward_pass!, backward_pass!, smooth!
export get_smoothed_trajectory, get_filtered_trajectory
export get_smoothed_state, get_smoothed_covariance

# Export multiple model utilities
export get_mode_probabilities, get_model_probabilities, get_model_estimates
export most_probable_mode, most_probable_model
export set_state!, get_joint_probabilities

# Export grid utilities
export generate_uniform_grid, find_closest_grid_point
export get_probability_mass, maximum_probability_state
export num_grid_points, get_grid_statistics
export is_within_bounds, compute_cell_size

# Export RBPF utilities (model interface - user implements these)
# Note: propagate_nonlinear, linear_dynamics_matrix, etc. are part of model interface

# Export marginal particle filter utilities
export get_marginal_estimate, get_marginal_variance, get_marginal_particles

end # module
