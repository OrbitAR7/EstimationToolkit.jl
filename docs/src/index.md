# EstimationToolkit.jl Documentation

```@meta
CurrentModule = EstimationToolkit
```

Documentation for **EstimationToolkit.jl**, a Julia package of state estimation filters that share one `predict!`/`update!` interface.

## Overview

The package has 36 filters (34 stable, 2 experimental) and 2 smoothers: Kalman-family, extended, sigma-point, ensemble, Gaussian-mixture, particle, particle-flow, multiple-model and grid-based. It's a research and teaching codebase (v0.9). Small states use StaticArrays, and model dimensions are type parameters, so mismatched models fail early.

## Quick Links

- [Getting Started](@ref getting_started)
- [Filter Catalog](@ref filters)
- [Examples](@ref examples)
- [API Reference](@ref api)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/OrbitAR7/EstimationToolkit.jl")
```

## Quick Example

```julia
using EstimationToolkit
using LinearAlgebra, StaticArrays

# Initialize Kalman filter
x0 = @SVector [0.0, 0.0, 1.0, 0.5]  # [x, y, vx, vy]
P0 = SMatrix{4,4}(1.0I)
kf = KalmanFilter(x0, P0)

# Define your dynamics and measurement models
# (see Getting Started for details)

# Filtering loop
predict!(kf, dynamics, nothing, dt)
update!(kf, measurement_model, observation)

x̂ = state_estimate(kf)
```

## Filter Categories

### Linear Filters
Optimal MMSE estimators for linear-Gaussian systems.

- Kalman Filter
- Information Filter  
- Square-Root variants
- Smoothers (RTS, SRIS)

### Nonlinear Approximations
For mildly to moderately nonlinear systems.

- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Cubature/Quadrature filters
- Sigma-point methods

### Sequential Monte Carlo
For arbitrary nonlinearity and non-Gaussian noise.

- Bootstrap Particle Filter
- Optimal proposal variants
- Rao-Blackwellized PF
- Particle flow methods

### Advanced Methods
Specialized filters for complex scenarios.

- Multiple model filters (IMM, GPB)
- Ensemble methods (EnKF, ETKF)
- Gaussian mixture filters
- Grid-based methods

## Getting Help

- Browse the [Examples](@ref examples) for typical usage patterns
- Check the [API Reference](@ref api) for detailed function signatures
- See individual filter documentation for algorithm details
- Open an issue on GitHub for bugs or feature requests

## Citation

If you use EstimationToolkit.jl in research, please cite:

```bibtex
@software{alshammari_estimationtoolkit,
  author = {Alshammari, Hamoud},
  title = {EstimationToolkit.jl},
  year = {2026},
  url = {https://github.com/OrbitAR7/EstimationToolkit.jl}
}
```

## License

MIT License. See [LICENSE](https://github.com/OrbitAR7/EstimationToolkit.jl/blob/main/LICENSE) for details.
