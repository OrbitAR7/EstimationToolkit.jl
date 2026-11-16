"""
Non-Gaussian orbit uncertainty: what happens to a Gaussian after a tracking gap.

A satellite in a 700 km, 60° orbit is known to 2 km and 2 m/s (1σ per axis) and is
then not tracked for two revolutions. The orbits in that initial uncertainty have
slightly different periods, so the uncertainty spreads to several hundred km along
the track, and because the track is curved the spread is curved too. A filter that
carries one mean and one covariance describes that region with an ellipsoid.

Two experiments:

1. `run_propagation()`: no measurements. The true distribution is a Monte Carlo
   cloud; each filter's predicted density is scored against it by the
   Kullback-Leibler divergence (0 = exact).
2. `run_tracking()`: after the gap, a ground radar measures range and range-rate
   for one pass. Monte Carlo over the initial error; each filter is scored by its
   position error and by whether its covariance is consistent with that error
   (NEES).

Filters: EKF, UKF, and the adaptive Gaussian mixture filter, which splits its
components during the gap.

Dynamics are point-mass gravity plus J2, state [x y z vx vy vz] in km and km/s,
inertial. Truth and filters use the same dynamics and the same integrator, so the
only thing being tested is how the uncertainty is represented.

Run:   julia --project examples/orbit_uncertainty.jl
Plots: see examples/plot_orbit_uncertainty.jl
"""

using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random
using Statistics
using Printf

const MU = 398600.4418            # km^3/s^2
const R_EARTH = 6378.137          # km
const J2 = 1.08262668e-3
const OMEGA_EARTH = 7.2921159e-5  # rad/s

# ---------------------------------------------------------------------------
# Dynamics: two-body + J2, fixed-step RK4
# ---------------------------------------------------------------------------

function state_derivative(x)
    r = SVector(x[1], x[2], x[3])
    rn = norm(r)
    k = 1.5 * J2 * MU * R_EARTH^2 / rn^5
    z2 = 5 * r[3]^2 / rn^2
    a = -MU / rn^3 * r + k * SVector(r[1] * (z2 - 1), r[2] * (z2 - 1), r[3] * (z2 - 3))
    return SVector(x[4], x[5], x[6], a[1], a[2], a[3])
end

function rk4_step(x, h)
    k1 = state_derivative(x)
    k2 = state_derivative(x + h / 2 * k1)
    k3 = state_derivative(x + h / 2 * k2)
    k4 = state_derivative(x + h * k3)
    return x + h / 6 * (k1 + 2k2 + 2k3 + k4)
end

"""Orbit dynamics integrated with RK4 substeps of at most `h` seconds. No process noise."""
struct J2Orbit <: AbstractDynamicsModel{6, 0, 6}
    h::Float64
end

function EstimationToolkit.propagate(m::J2Orbit, x, u, dt)
    n = max(1, ceil(Int, dt / m.h - 1e-9))
    h = dt / n
    for _ in 1:n
        x = rk4_step(x, h)
    end
    return x
end

EstimationToolkit.process_noise_cov(::J2Orbit, dt) = @SMatrix zeros(6, 6)

# ---------------------------------------------------------------------------
# Measurements: range and range-rate from a ground station
# ---------------------------------------------------------------------------

struct RadarRangeRate <: AbstractMeasurementModel{6, 2, 2}
    r_station::SVector{3, Float64}   # inertial, km
    v_station::SVector{3, Float64}   # inertial, km/s
    σ_range::Float64                 # km
    σ_rate::Float64                  # km/s
end

function EstimationToolkit.measure(m::RadarRangeRate, x)
    dr = SVector(x[1], x[2], x[3]) - m.r_station
    dv = SVector(x[4], x[5], x[6]) - m.v_station
    ρ = norm(dr)
    return SVector(ρ, dot(dr, dv) / ρ)
end

EstimationToolkit.measurement_noise_cov(m::RadarRangeRate) =
    @SMatrix [m.σ_range^2 0.0; 0.0 m.σ_rate^2]

"""Inertial position and velocity of a station on a spherical, uniformly rotating Earth."""
function station_state(t, lat, lon0)
    θ = lon0 + OMEGA_EARTH * t
    r = R_EARTH * SVector(cos(lat) * cos(θ), cos(lat) * sin(θ), sin(lat))
    return r, OMEGA_EARTH * SVector(-r[2], r[1], 0.0)
end

function elevation(x, r_station)
    d = SVector(x[1], x[2], x[3]) - r_station
    return asin(dot(d, r_station) / (norm(d) * norm(r_station)))
end

# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

function scenario(; altitude = 700.0, inclination = deg2rad(60.0), σ_pos = 2.0, σ_vel = 0.002)
    a = R_EARTH + altitude
    v = sqrt(MU / a)
    x0 = SVector(a, 0.0, 0.0, 0.0, v * cos(inclination), v * sin(inclination))
    P0 = SMatrix{6, 6}(Diagonal(SVector(σ_pos^2, σ_pos^2, σ_pos^2, σ_vel^2, σ_vel^2, σ_vel^2)))
    period = 2π * sqrt(a^3 / MU)
    return (; x0, P0, period, dyn = J2Orbit(30.0))
end

# UKF with sigma points at √6 σ (α = 1). The package default α = 1e-3 keeps them so
# close to the mean that the UKF behaves like an EKF with a second-order mean term.
make_ukf(x0, P0) = UnscentedKalmanFilter(x0, P0; α = 1.0, β = 2.0, κ = 0.0)

gaussian_entropy(P) = 0.5 * logdet(2π * ℯ * Matrix(P))

"""Mean of (x - x̂)' P⁻¹ (x - x̂) over the samples, with the filter's own x̂ and P."""
function mean_mahalanobis(f, samples)
    x̂ = state_estimate(f)
    C = cholesky(Symmetric(Matrix(covariance(f))))
    return mean(samples) do s
        e = Vector(s - x̂)
        dot(e, C \ e)
    end
end

# ---------------------------------------------------------------------------
# 1. Propagation only
# ---------------------------------------------------------------------------

"""
    run_propagation(; orbits=2, dt=60, n_samples=20_000, component_limits=(15, 50, 150), ...)

Propagate the initial Gaussian for `orbits` revolutions with no measurements.

The score is the Kullback-Leibler divergence from the true density p to the
filter's density q, KL(p‖q) = -H(p) - E_p[log q]. The flow of these dynamics
preserves phase-space volume (Liouville), so the entropy H(p) of the true
distribution stays at its initial value, which is known in closed form; E_p[log q]
is the average of the filter's log-density over the Monte Carlo samples.
"""
function run_propagation(; orbits = 2.0, dt = 60.0, n_samples = 20_000,
                         component_limits = (15, 50, 150), split_tol = 0.05,
                         seed = 2026, verbose = true)
    sc = scenario()
    (; x0, P0, period, dyn) = sc
    K = round(Int, orbits * period / dt)

    rng = MersenneTwister(seed)
    L = cholesky(Symmetric(Matrix(P0))).L
    samples = [x0 + SVector{6}(L * randn(rng, 6)) for _ in 1:n_samples]
    H0 = gaussian_entropy(P0)

    ekf = ExtendedKalmanFilter(x0, P0)
    ukf = make_ukf(x0, P0)
    mixtures = [AdaptiveGaussianMixtureFilter(x0, P0; split_tol, max_components = m)
                for m in component_limits]

    labels = ["EKF", "UKF", ["mixture ≤$m" for m in component_limits]...]
    record_every = max(1, round(Int, period / dt / 8))       # eight times per revolution
    times = Float64[]
    kl = [Float64[] for _ in labels]
    ncomp = [Int[] for _ in mixtures]

    gaussian_score(f) = mean(gaussian_logpdf(s, state_estimate(f), covariance(f)) for s in samples)

    function record!(t)
        push!(times, t)
        push!(kl[1], -H0 - gaussian_score(ekf))
        push!(kl[2], -H0 - gaussian_score(ukf))
        for (i, f) in enumerate(mixtures)
            push!(kl[2 + i], -H0 - mean(mixture_logpdf(f, samples)))
            push!(ncomp[i], num_components(f))
        end
    end

    record!(0.0)
    for k in 1:K
        samples = [propagate(dyn, s, nothing, dt) for s in samples]
        predict!(ekf, dyn, nothing, dt)
        predict!(ukf, dyn, nothing, dt)
        for f in mixtures
            predict!(f, dyn, nothing, dt)
        end
        (k % record_every == 0 || k == K) && record!(k * dt)
    end

    mc_mean = mean(samples)
    mc_cov = cov(reduce(hcat, samples)')
    filters = (ekf, ukf, mixtures...)

    if verbose
        println("Propagation for $(orbits) revolutions, no measurements ($(n_samples) Monte Carlo samples)")
        println("\nKL divergence from the true distribution [nats], 0 = exact")
        @printf("%-9s", "revs")
        foreach(l -> @printf("%14s", l), labels)
        println()
        for i in eachindex(times)
            (i - 1) % 4 == 0 || i == length(times) || continue
            @printf("%-9.2f", times[i] / period)
            foreach(j -> @printf("%14.2f", kl[j][i]), eachindex(labels))
            println()
        end
        println("\nAt the end (d² = squared Mahalanobis distance of the true samples from the")
        println("filter's mean under the filter's covariance; 6 on average if the covariance fits):")
        @printf("%-14s %12s %22s %14s\n", "", "components", "mean position error", "mean d²")
        for (l, f) in zip(labels, filters)
            n = f isa AdaptiveGaussianMixtureFilter ? num_components(f) : 1
            @printf("%-14s %12d %19.2f km %14.1f\n", l, n,
                    norm((state_estimate(f) - mc_mean)[SOneTo(3)]), mean_mahalanobis(f, samples))
        end
        println("(position error is relative to the Monte Carlo mean)")
    end

    return (; sc, times, kl, labels, ncomp, component_limits, samples, mc_mean, mc_cov,
            ekf, ukf, mixtures)
end

# ---------------------------------------------------------------------------
# 2. Tracking pass after the gap
# ---------------------------------------------------------------------------

"""
    run_tracking(; trials=200, gap_orbits=2, ...)

Monte Carlo over the initial error. Each trial: draw the true state from N(x0, P0),
coast for `gap_orbits` revolutions, then process one radar pass (range and
range-rate every `dt_meas` seconds while the satellite is above 10° elevation).

Reported per filter, as a function of time into the pass:
- median position error
- median NEES, e' P⁻¹ e with the filter's own covariance. For a consistent
  6-state filter it is χ²(6): median 5.35, and 99 % of trials below 16.81.
"""
function run_tracking(; trials = 200, gap_orbits = 2.0, dt = 60.0, dt_meas = 10.0,
                      pass_duration = 480.0, σ_range = 0.010, σ_rate = 1e-5,
                      split_tol = 0.05, max_components = 50, seed = 2026, verbose = true)
    sc = scenario()
    (; x0, P0, period, dyn) = sc
    K_gap = round(Int, gap_orbits * period / dt)
    t_gap = K_gap * dt
    K_pass = round(Int, pass_duration / dt_meas)

    # Put the station 3° north of where the reference orbit is in mid-pass.
    x_mid = propagate(dyn, x0, nothing, t_gap + pass_duration / 2)
    lat = asin(x_mid[3] / norm(x_mid[SOneTo(3)])) + deg2rad(3.0)
    lon0 = atan(x_mid[2], x_mid[1]) - OMEGA_EARTH * (t_gap + pass_duration / 2)

    names = (:ekf, :ukf, :mixture)
    nees = Dict(n => fill(NaN, trials, K_pass) for n in names)
    pos_err = Dict(n => fill(NaN, trials, K_pass) for n in names)
    ncomp = zeros(Int, trials, K_pass)
    n_meas = zeros(Int, trials)

    rng = MersenneTwister(seed)
    L = cholesky(Symmetric(Matrix(P0))).L

    for trial in 1:trials
        x_true = x0 + SVector{6}(L * randn(rng, 6))
        filters = (ekf = ExtendedKalmanFilter(x0, P0),
                   ukf = make_ukf(x0, P0),
                   mixture = AdaptiveGaussianMixtureFilter(x0, P0; split_tol, max_components))

        for _ in 1:K_gap
            x_true = propagate(dyn, x_true, nothing, dt)
            foreach(f -> predict!(f, dyn, nothing, dt), filters)
        end

        for k in 1:K_pass
            t = t_gap + k * dt_meas
            x_true = propagate(dyn, x_true, nothing, dt_meas)
            foreach(f -> predict!(f, dyn, nothing, dt_meas), filters)

            r_s, v_s = station_state(t, lat, lon0)
            if elevation(x_true, r_s) > deg2rad(10.0)
                meas = RadarRangeRate(r_s, v_s, σ_range, σ_rate)
                y = measure(meas, x_true) + SVector(σ_range * randn(rng), σ_rate * randn(rng))
                foreach(f -> update!(f, meas, y), filters)
                n_meas[trial] += 1
            end

            for n in names
                e = Vector(state_estimate(filters[n]) - x_true)
                P = Symmetric(Matrix(covariance(filters[n])))
                nees[n][trial, k] = dot(e, P \ e)
                pos_err[n][trial, k] = norm(e[1:3])
            end
            ncomp[trial, k] = num_components(filters.mixture)
        end
    end

    times = dt_meas .* (1:K_pass)
    χ2_99 = 16.81       # 99th percentile of χ²(6)

    if verbose
        println("Radar pass after a $(gap_orbits)-revolution gap: $(trials) trials, ",
                "range $(round(Int, 1e3σ_range)) m, range-rate $(round(1e5σ_rate, digits = 1)) cm/s, ",
                "every $(round(Int, dt_meas)) s")
        @printf("measurements per trial: %d to %d\n\n", minimum(n_meas), maximum(n_meas))
        println("median position error [km]              median NEES (5.35 if consistent)    trials with NEES < 16.81 (99 % if consistent)")
        @printf("%-8s %9s %9s %9s    %10s %9s %9s    %8s %8s %8s   %s\n", "t [s]",
                "EKF", "UKF", "mixture", "EKF", "UKF", "mixture", "EKF", "UKF", "mixture", "components")
        for k in unique(clamp.(round.(Int, (10, 30, 60, 120, 240, 480) ./ dt_meas), 1, K_pass))
            @printf("%-8d", times[k])
            foreach(n -> @printf(" %9.3f", median(pos_err[n][:, k])), names)
            print("   ")
            foreach(n -> @printf(" %10.1f", median(nees[n][:, k])), names)
            print("   ")
            foreach(n -> @printf(" %7.0f%%", 100 * mean(nees[n][:, k] .< χ2_99)), names)
            @printf("   %d\n", round(Int, median(ncomp[:, k])))
        end
    end

    return (; sc, times, nees, pos_err, ncomp, n_meas, χ2_99, names)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_propagation()
    println()
    run_tracking()
end
