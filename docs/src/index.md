# EstimationToolkit.jl Documentation

```@meta
CurrentModule = EstimationToolkit
```

Welcome to the documentation for **EstimationToolkit.jl** — a comprehensive Julia package for state estimation and filtering.

## Overview

EstimationToolkit.jl provides production-ready implementations of 35+ estimation algorithms, from classical Kalman filters to advanced particle methods. Built for aerospace applications but applicable across robotics, navigation, and signal processing domains.

## Key Features

- **Comprehensive coverage** — Complete estimation algorithm taxonomy
- **Production quality** — Numerically stable, extensively tested
- **Type safety** — Compile-time dimension checking
- **Performance** — Efficient StaticArrays-based implementations
- **Clear documentation** — Every filter includes mathematical foundations

## Quick Links

- [Getting Started](@ref getting_started)
- [Filter Catalog](@ref filters)
- [Examples](@ref examples)
- [API Reference](@ref api)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/EstimationToolkit.jl")
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
@software{estimationtoolkit2026,
  author = {Alshammari, Hamoud},
  title = {EstimationToolkit.jl: Comprehensive State Estimation for Julia},
  year = {2026},
  url = {https://github.com/yourusername/EstimationToolkit.jl}
}
```

## License

MIT License. See [LICENSE](https://github.com/yourusername/EstimationToolkit.jl/blob/main/LICENSE) for details.
