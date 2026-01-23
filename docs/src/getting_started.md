# [Getting Started](@id getting_started)

This guide will help you get up and running with EstimationToolkit.jl.

## Installation

Install the package from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/EstimationToolkit.jl")
```

## Basic Concepts

EstimationToolkit.jl is built around three core concepts:

1. **Filters** — Objects that maintain state estimates
2. **Dynamics Models** — Describe how the system evolves over time
3. **Measurement Models** — Describe how observations relate to state

## Your First Filter

Let's implement a simple 2D tracking problem.

### Step 1: Define the Dynamics

```julia
using EstimationToolkit
using LinearAlgebra
using StaticArrays

# Constant velocity model
struct ConstantVelocity2D <: AbstractDynamicsModel{4, 0, 4}
    q::Float64  # Process noise intensity
end

# Propagation function
function EstimationToolkit.propagate(model::ConstantVelocity2D, x, u, dt)
    F = @SMatrix [1.0 0.0  dt 0.0;
                  0.0 1.0 0.0  dt;
                  0.0 0.0 1.0 0.0;
                  0.0 0.0 0.0 1.0]
    return F * x
end

# Process noise covariance
EstimationToolkit.process_noise_cov(m::ConstantVelocity2D, dt) =
    m.q * @SMatrix [dt^3/3 0 dt^2/2 0;
                    0 dt^3/3 0 dt^2/2;
                    dt^2/2 0 dt 0;
                    0 dt^2/2 0 dt]
```

### Step 2: Define the Measurement Model

```julia
# Position-only measurements
struct PositionMeasurement2D <: AbstractMeasurementModel{4, 2, 2}
    r::Float64  # Measurement noise std
end

EstimationToolkit.measure(m::PositionMeasurement2D, x) = x[1:2]

EstimationToolkit.measurement_noise_cov(m::PositionMeasurement2D) =
    m.r^2 * @SMatrix [1.0 0.0; 0.0 1.0]
```

### Step 3: Initialize and Run the Filter

```julia
# Initial state: [x=0, y=0, vx=1, vy=0.5]
x0 = @SVector [0.0, 0.0, 1.0, 0.5]
P0 = SMatrix{4,4}(1.0I)

# Create filter
kf = KalmanFilter(x0, P0)

# Create models
dynamics = ConstantVelocity2D(0.1)
measurement = PositionMeasurement2D(0.5)

# Simulate measurements (placeholder)
measurements = [...]  # Your sensor data here

# Filter loop
dt = 0.1
for z in measurements
    predict!(kf, dynamics, nothing, dt)
    update!(kf, measurement, z)
    
    # Get current estimate
    x̂ = state_estimate(kf)
    P = covariance(kf)
    
    println("Estimate: $x̂")
end
```

## Choosing a Filter

Here's a quick guide to filter selection:

| System Type | Recommended Filter | Notes |
|-------------|-------------------|-------|
| Linear + Gaussian noise | `KalmanFilter` | Optimal MMSE estimator |
| Mildly nonlinear | `ExtendedKalmanFilter` | First-order linearization |
| Moderately nonlinear | `UnscentedKalmanFilter` | Better accuracy, no Jacobians |
| Strongly nonlinear | `BootstrapParticleFilter` | Handles arbitrary nonlinearity |
| Multi-modal | `GaussianSumFilter` or `PointMassFilter` | Represents multiple hypotheses |
| High-dimensional (100+) | `EnsembleKalmanFilter` | Scalable Monte Carlo |

## Next Steps

- Check out complete [Examples](@ref examples)
- Browse the [Filter Catalog](@ref filters) for all available filters
- Read individual filter documentation for algorithm details
- See [API Reference](@ref api) for complete function signatures

## Common Patterns

### Accessing Filter State

```julia
x̂ = state_estimate(filter)  # Current state estimate
P = covariance(filter)  # Current covariance matrix
```

### Running Multiple Steps

```julia
for k in 1:n_steps
    predict!(filter, dynamics, nothing, dt)
    update!(filter, measurement_model, measurements[k])
end
```

### Handling Control Inputs

```julia
# If your dynamics include control
u = @SVector [thrust, steering_angle]
predict!(filter, dynamics, u, dt)
```

### Diagnostics

```julia
# Check filter consistency
nis_value = nis(filter)  # Normalized Innovation Squared
nees_value = nees(filter, x_true)  # Normalized Estimation Error Squared
```

## Tips for Success

1. **Use StaticArrays** for small states (< 100 dimensions) for best performance
2. **Check dimensions** — Julia's type system will catch mismatches at compile time
3. **Test on linear systems** first — validate your implementation
4. **Monitor diagnostics** — NIS/NEES tests reveal inconsistencies
5. **Start simple** — Begin with KF/EKF before trying advanced methods
