"""
Navigating a near-rectilinear halo orbit around the Moon.

A spacecraft is on the 9:2 synodic resonant NRHO of the Earth-Moon system (the
orbit chosen for the lunar Gateway): 6.56 days per revolution, 71 000 km from the
Moon at apolune and 3 200 km at perilune, which it passes at 1.7 km/s. It is
tracked from one ground station by range and range-rate.

The question is the same as in `orbit_uncertainty.jl`, in a different regime: when
is a single Gaussian an adequate description of what is known about the state?

Two experiments:

1. `run_coast()`: no measurements for three revolutions. Each filter's predicted
   density is scored against a Monte Carlo cloud by the Kullback-Leibler
   divergence (0 = exact).
2. `run_tracking()`: two revolutions of tracking with a 3.5-day outage that spans
   a perilune passage. Monte Carlo over the initial error and the noise; position
   error and NEES for each filter.

Filters: EKF, UKF and the adaptive Gaussian mixture filter.

The dynamics are the circular restricted three-body problem, from
OrbitalMechanics.jl (https://github.com/OrbitAR7/orbital-mechanics-jl), which also
supplies the orbit itself and the state transition matrix that the EKF uses as its
Jacobian. States are in the rotating frame, in that package's nondimensional
units; results are printed in km and days.

This example needs OrbitalMechanics in the environment as well:

    julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path=".");
              Pkg.add(url="https://github.com/OrbitAR7/orbital-mechanics-jl");
              Pkg.add("StaticArrays")' -L examples/nrho_navigation.jl -e 'run_coast(); run_tracking()'

Plots: see examples/plot_nrho_navigation.jl
"""

using EstimationToolkit
using OrbitalMechanics
using LinearAlgebra
using StaticArrays
using Random
using Statistics
using Printf

const SYSTEM = EARTH_MOON
const LU = SYSTEM.l_star                 # km per length unit
const TU = SYSTEM.t_star                 # s per time unit
const VU = LU / TU                       # km/s per velocity unit
const DAY = 86400.0 / TU                 # one day in time units
const SYNODIC_MONTH_DAYS = 29.530588853

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

"""
CR3BP dynamics with white-noise acceleration of spectral density `q`
(nondimensional). `q = 0` is the deterministic flow.
"""
struct CR3BPDynamics <: AbstractDynamicsModel{6, 0, 6}
    μ::Float64
    q::Float64
end

EstimationToolkit.propagate(m::CR3BPDynamics, x, u, dt) =
    integrate(y -> cr3bp_eom(y, m.μ), SVector{6, Float64}(x), dt; rtol = 1e-11, atol = 1e-13)

# The EKF's Jacobian is the state transition matrix, integrated from the
# variational equations alongside the state.
EstimationToolkit.jacobian_x(m::CR3BPDynamics, x::SVector, u, dt) =
    propagate_cr3bp(SYSTEM, x, dt; stm = true, rtol = 1e-11, atol = 1e-13)[2]

function EstimationToolkit.process_noise_cov(m::CR3BPDynamics, dt)
    I3 = SMatrix{3, 3}(1.0I)
    return m.q * [dt^3/3*I3 dt^2/2*I3; dt^2/2*I3 dt*I3]
end

"""Range and range-rate from a ground station whose state in the rotating frame is (r, v)."""
struct StationRangeRate <: AbstractMeasurementModel{6, 2, 2}
    r::SVector{3, Float64}
    v::SVector{3, Float64}
    σ_range::Float64
    σ_rate::Float64
end

# The distance between two points does not depend on the frame it is computed in,
# so neither does its rate: no transformation to an inertial frame is needed.
function EstimationToolkit.measure(m::StationRangeRate, x)
    dr = SVector(x[1], x[2], x[3]) - m.r
    dv = SVector(x[4], x[5], x[6]) - m.v
    ρ = norm(dr)
    return SVector(ρ, dot(dr, dv) / ρ)
end

EstimationToolkit.measurement_noise_cov(m::StationRangeRate) =
    @SMatrix [m.σ_range^2 0.0; 0.0 m.σ_rate^2]

# One station on the equator of an Earth whose equator lies in the Earth-Moon
# plane. The Earth turns once per sidereal day; seen from the rotating frame, which
# itself turns once per month, a little slower.
const EARTH_RADIUS = 6378.137 / LU
const EARTH_RATE = 2π / (86164.1 / TU) - 1.0

function station_state(t)
    θ = EARTH_RATE * t
    r = SVector(-SYSTEM.μ + EARTH_RADIUS * cos(θ), EARTH_RADIUS * sin(θ), 0.0)
    v = EARTH_RADIUS * EARTH_RATE * SVector(-sin(θ), cos(θ), 0.0)
    return r, v
end

"""True if the spacecraft is more than 10° above the station's horizon."""
function in_view(x, t)
    r, _ = station_state(t)
    up = r - primary_position(SYSTEM)
    d = SVector(x[1], x[2], x[3]) - r
    return dot(d, up) / (norm(d) * norm(up)) > sind(10.0)
end

# ---------------------------------------------------------------------------
# The orbit
# ---------------------------------------------------------------------------

"""
The 9:2 NRHO, computed from the mass ratio: follow the L2 southern halo family
from its bifurcation and take the member with nine revolutions in two synodic
months. Its state `x0` is at apolune.
"""
function gateway_nrho()
    family = halo_family(SYSTEM, :L2; branch = :southern, n = 600, ds_max = 8e-3,
                         stop = o -> secondary_distance_range(o)[1] * LU < 2100)
    T = (2 / 9) * SYNODIC_MONTH_DAYS * DAY
    k = findfirst(i -> (family[i].period - T) * (family[i+1].period - T) <= 0, 1:length(family)-1)
    return orbit_with_period(SYSTEM, family[k], family[k+1], T)
end

initial_covariance(σ_pos_km, σ_vel_ms) =
    SMatrix{6, 6}(Diagonal(SVector(fill((σ_pos_km / LU)^2, 3)..., fill((σ_vel_ms * 1e-3 / VU)^2, 3)...)))

# UKF with sigma points at √6 σ, as in orbit_uncertainty.jl
make_ukf(x0, P0) = UnscentedKalmanFilter(x0, P0; α = 1.0, β = 2.0, κ = 0.0)

moon_distance_km(x) = norm(SVector(x[1], x[2], x[3]) - secondary_position(SYSTEM)) * LU

# ---------------------------------------------------------------------------
# 1. Coast
# ---------------------------------------------------------------------------

"""
    run_coast(; revolutions=3, σ_pos_km=1, σ_vel_ms=0.01, n_samples=4000, ...)

Propagate a Gaussian with the given 1σ position and velocity uncertainty (per
axis) from apolune, with no measurements and no process noise.

Scoring is as in `orbit_uncertainty.jl`: KL(p‖q) = -H(p) - E_p[log q], with the
entropy H(p) of the true distribution constant because the flow preserves
phase-space volume (the CR3BP vector field has zero divergence in these
coordinates), and the expectation taken over the Monte Carlo samples.
"""
function run_coast(; revolutions = 3.0, dt_minutes = 15.0, σ_pos_km = 1.0, σ_vel_ms = 0.01,
                   n_samples = 3000, component_limits = (25, 100), split_tol = 0.05,
                   seed = 2026, verbose = true)
    orbit = gateway_nrho()
    dyn = CR3BPDynamics(SYSTEM.μ, 0.0)
    dt = dt_minutes * 60 / TU
    K = round(Int, revolutions * orbit.period / dt)
    x0, P0 = orbit.x0, initial_covariance(σ_pos_km, σ_vel_ms)

    rng = MersenneTwister(seed)
    L = cholesky(Symmetric(Matrix(P0))).L
    samples = [x0 + SVector{6}(L * randn(rng, 6)) for _ in 1:n_samples]
    H0 = 0.5 * logdet(2π * ℯ * Matrix(P0))

    ekf = ExtendedKalmanFilter(x0, P0)
    ukf = make_ukf(x0, P0)
    mixtures = [AdaptiveGaussianMixtureFilter(x0, P0; split_tol, max_components = m) for m in component_limits]
    labels = ["EKF", "UKF", ["mixture ≤$m" for m in component_limits]...]

    function gaussian_kl(f)
        x̂ = state_estimate(f)
        C = cholesky(Symmetric(Matrix(covariance(f))))
        score = mean(samples) do s
            e = Vector(s - x̂)
            -0.5 * (dot(e, C \ e) + 6 * log(2π) + logdet(C))
        end
        return -H0 - score
    end

    days = zeros(K)
    kl = [zeros(K) for _ in labels]
    moon_km = zeros(K)
    ncomp = [zeros(Int, K) for _ in mixtures]
    x_ref = x0

    for k in 1:K
        samples = [propagate(dyn, s, nothing, dt) for s in samples]
        x_ref = propagate(dyn, x_ref, nothing, dt)
        predict!(ekf, dyn, nothing, dt)
        predict!(ukf, dyn, nothing, dt)
        foreach(f -> predict!(f, dyn, nothing, dt), mixtures)

        days[k] = k * dt / DAY
        moon_km[k] = moon_distance_km(x_ref)
        kl[1][k] = gaussian_kl(ekf)
        kl[2][k] = gaussian_kl(ukf)
        for (i, f) in enumerate(mixtures)
            kl[2+i][k] = -H0 - mean(mixture_logpdf(f, samples))
            ncomp[i][k] = num_components(f)
        end
    end

    period_days = orbit.period / DAY
    perilunes = [(n - 0.5) * period_days for n in 1:floor(Int, revolutions + 0.5)]
    apolunes = [n * period_days for n in 1:floor(Int, revolutions)]

    if verbose
        @printf("Coast for %.0f revolutions from apolune, %.0f km and %.0f cm/s (1σ per axis), %d samples\n",
                revolutions, σ_pos_km, 100σ_vel_ms, n_samples)
        println("\nKL divergence from the true distribution [nats], 0 = exact")
        @printf("%-26s", "")
        foreach(l -> @printf("%14s", l), labels)
        println()
        # worst value within six hours of each perilune, and the value at each apolune
        events = sort(vcat([(t, "perilune") for t in perilunes], [(t, "apolune") for t in apolunes]))
        for (t, kind) in events
            @printf("%-9s day %5.2f        ", kind, t)
            for j in eachindex(labels)
                if kind == "perilune"
                    window = findall(d -> abs(d - t) <= 0.25, days)
                    @printf("%14.2f", maximum(kl[j][window]))
                else
                    @printf("%14.2f", kl[j][argmin(abs.(days .- t))])
                end
            end
            println()
        end
        println("(perilune rows: largest value within six hours of perilune)")
        # How long the EKF is wrong for: time around each perilune with KL above 1 nat
        step_hours = dt_minutes / 60
        print("\nEKF more than 1 nat from the truth for")
        for (i, t) in enumerate(perilunes)
            window = findall(d -> abs(d - t) <= 1.0, days)
            @printf("%s %.1f h", i == 1 ? "" : ",", step_hours * count(>(1.0), kl[1][window]))
        end
        println(" around the successive perilunes; never elsewhere")
        @printf("largest EKF value more than a day from any perilune: %.3f nats\n",
                maximum(kl[1][[all(t -> abs(d - t) > 1.0, perilunes) for d in days]]))
    end

    return (; orbit, days, kl, labels, moon_km, ncomp, component_limits, perilunes, apolunes, period_days)
end

# ---------------------------------------------------------------------------
# 2. Tracking with an outage across perilune
# ---------------------------------------------------------------------------

"""
    run_tracking(; trials=100, revolutions=2, outage_days=(7.0, 10.5), ...)

Monte Carlo of two revolutions of tracking from one ground station: range (10 m)
and range-rate (1 mm/s) every 10 minutes whenever the spacecraft is above 10°
elevation, except during `outage_days`, when the station is down. The second
perilune (day 9.84) falls inside the outage.

Truth and filters share the dynamics, including unmodelled accelerations at the
level of 1e-7 m/s² (white, the same intensity in the truth and in the filters'
process noise). The initial error is 10 km and 10 cm/s (1σ per axis).

For a consistent 6-state filter the NEES is χ²(6): median 5.35, and 99 % of trials
below 16.81.
"""
function run_tracking(; trials = 100, revolutions = 2.0, dt_minutes = 10.0, outage_days = (7.0, 10.5),
                      σ_pos_km = 10.0, σ_vel_ms = 0.1, σ_range_m = 10.0, σ_rate_mms = 1.0,
                      σ_accel = 1e-10, split_tol = 0.05, max_components = 20, seed = 2026, verbose = true)
    orbit = gateway_nrho()
    dt = dt_minutes * 60 / TU
    K = round(Int, revolutions * orbit.period / dt)

    # Acceleration noise: σ_accel [km/s²] held over one step, as a spectral density
    q = (σ_accel / (LU / TU^2))^2 * dt
    dyn = CR3BPDynamics(SYSTEM.μ, q)
    LQ = cholesky(Symmetric(Matrix(process_noise_cov(dyn, dt)))).L

    x0, P0 = orbit.x0, initial_covariance(σ_pos_km, σ_vel_ms)
    L0 = cholesky(Symmetric(Matrix(P0))).L
    σ_range, σ_rate = σ_range_m * 1e-3 / LU, σ_rate_mms * 1e-6 / VU

    names = (:ekf, :ukf, :mixture)
    nees = Dict(n => zeros(trials, K) for n in names)
    pos_err = Dict(n => zeros(trials, K) for n in names)
    ncomp = zeros(Int, trials, K)
    tracked = falses(K)                 # was a measurement processed at step k (first trial)
    runtime = Dict(n => 0.0 for n in names)
    rng = MersenneTwister(seed)

    for trial in 1:trials
        x_true = x0 + SVector{6}(L0 * randn(rng, 6))
        filters = (ekf = ExtendedKalmanFilter(x0, P0), ukf = make_ukf(x0, P0),
                   mixture = AdaptiveGaussianMixtureFilter(x0, P0; split_tol, max_components))
        for k in 1:K
            t = k * dt
            x_true = propagate(dyn, x_true, nothing, dt) + SVector{6}(LQ * randn(rng, 6))
            for n in names
                runtime[n] += @elapsed predict!(filters[n], dyn, nothing, dt)
            end

            if in_view(x_true, t) && !(outage_days[1] <= t / DAY < outage_days[2])
                r_s, v_s = station_state(t)
                meas = StationRangeRate(r_s, v_s, σ_range, σ_rate)
                y = measure(meas, x_true) + SVector(σ_range * randn(rng), σ_rate * randn(rng))
                for n in names
                    runtime[n] += @elapsed update!(filters[n], meas, y)
                end
                trial == 1 && (tracked[k] = true)
            end

            for n in names
                e = Vector(state_estimate(filters[n]) - x_true)
                P = Symmetric(Matrix(covariance(filters[n])))
                nees[n][trial, k] = dot(e, P \ e)
                pos_err[n][trial, k] = norm(e[1:3]) * LU
            end
            ncomp[trial, k] = num_components(filters.mixture)
        end
    end

    days = (1:K) .* (dt / DAY)
    period_days = orbit.period / DAY
    perilunes = [(n - 0.5) * period_days for n in 1:floor(Int, revolutions + 0.5)]
    χ2_99 = 16.81

    if verbose
        @printf("Tracking for %.0f revolutions: %d trials, range %.0f m, range-rate %.0f mm/s every %.0f min, outage on days %.1f to %.1f\n",
                revolutions, trials, σ_range_m, σ_rate_mms, dt_minutes, outage_days...)
        @printf("station in view for %.0f %% of the time\n\n", 100 * (count(tracked) / K) / (1 - (outage_days[2] - outage_days[1]) / days[end]))
        println("                                   median position error [km]        median NEES (5.35)       NEES < 16.81 (99 %)")
        @printf("%-34s %8s %8s %8s   %8s %7s %8s   %6s %6s %8s\n", "", "EKF", "UKF", "mixture",
                "EKF", "UKF", "mixture", "EKF", "UKF", "mixture")
        k_peak = (w = findall(d -> abs(d - perilunes[2]) <= 0.25, days); w[argmax([median(nees[:ekf][:, k]) for k in w])])
        rows = (("before the outage, day 6.9", argmin(abs.(days .- 6.9))),
                (@sprintf("perilune in the outage, day %.2f", days[k_peak]), k_peak),
                ("6 h after perilune", argmin(abs.(days .- (perilunes[2] + 0.25)))),
                ("tracking again, day 11.0", argmin(abs.(days .- 11.0))),
                (@sprintf("end, day %.1f", days[end]), K))
        for (label, k) in rows
            @printf("%-34s", label)
            foreach(n -> @printf(" %8.3f", median(pos_err[n][:, k])), names)
            print("  ")
            foreach(n -> @printf(" %8.1f", median(nees[n][:, k])), names)
            print("  ")
            foreach(n -> @printf(" %6.0f%%", 100 * mean(nees[n][:, k] .< χ2_99)), names)
            println()
        end
        @printf("\nmixture components: 1 until day %.2f, at most %d\n",
                days[something(findfirst(k -> maximum(ncomp[:, k]) > 1, 1:K), K)], maximum(ncomp))
        @printf("filter time for all trials [s]: EKF %.1f, UKF %.1f, mixture %.1f\n",
                runtime[:ekf], runtime[:ukf], runtime[:mixture])
    end

    return (; orbit, days, nees, pos_err, ncomp, tracked, names, χ2_99, perilunes, outage_days, runtime)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_coast()
    println()
    run_tracking()
end
