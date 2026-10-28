"""
Maneuver detection in LEO orbit determination: IMM and GPB2 vs. single EKFs.

A satellite in a 500 km circular orbit does an unannounced along-track burn
(0.1 m/s² for 300 s, Δv = 30 m/s) partway through the run. It is tracked by
range and range-rate from two geostationary relays 180° apart (TDRS-style),
so one of them is always in view.

Filters compared:
- EKF, ballistic process noise: right before the burn, badly wrong after it
- EKF, maneuver-level process noise: survives the burn, noisy the rest of the time
- IMM with both models (EKF per mode)
- GPB2 with both models

Everything is planar (equatorial orbit, state [x, y, vx, vy] in km and km/s)
to keep the example short. Point-mass gravity only; the filters use the same
gravity model as the truth, so the burn is the only model error.

Run:   julia --project examples/orbit_maneuver_imm.jl
Plots: see examples/plot_orbit_maneuver.jl
"""

using EstimationToolkit
using LinearAlgebra
using StaticArrays
using Random
using Printf

const MU = 398600.4418        # km^3/s^2
const R_EARTH = 6378.137      # km
const R_GEO = 42164.0         # km

# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

twobody(x, a_extra) = SVector(x[3], x[4],
                              -MU * x[1] / norm(SVector(x[1], x[2]))^3 + a_extra[1],
                              -MU * x[2] / norm(SVector(x[1], x[2]))^3 + a_extra[2])

function rk4(x, dt, a_extra = SVector(0.0, 0.0))
    k1 = twobody(x, a_extra)
    k2 = twobody(x + dt / 2 * k1, a_extra)
    k3 = twobody(x + dt / 2 * k2, a_extra)
    k4 = twobody(x + dt * k3, a_extra)
    return x + dt / 6 * (k1 + 2k2 + 2k3 + k4)
end

"""Two-body dynamics with white-noise acceleration of spectral density `q` [km²/s³]."""
struct TwoBodyWNA <: AbstractDynamicsModel{4, 0, 4}
    q::Float64
end

EstimationToolkit.propagate(m::TwoBodyWNA, x, u, dt) = rk4(x, dt)

function EstimationToolkit.process_noise_cov(m::TwoBodyWNA, dt)
    a, b, c = dt^3 / 3, dt^2 / 2, dt
    return m.q * @SMatrix [a 0 b 0;
                           0 a 0 b;
                           b 0 c 0;
                           0 b 0 c]
end

# ---------------------------------------------------------------------------
# Measurements: range and range-rate to a relay
# ---------------------------------------------------------------------------

struct RelayRangeRate <: AbstractMeasurementModel{4, 2, 2}
    r_relay::SVector{2, Float64}
    v_relay::SVector{2, Float64}
    σ_range::Float64       # km
    σ_rate::Float64        # km/s
end

function EstimationToolkit.measure(m::RelayRangeRate, x)
    dr = SVector(x[1], x[2]) - m.r_relay
    dv = SVector(x[3], x[4]) - m.v_relay
    ρ = norm(dr)
    return SVector(ρ, dot(dr, dv) / ρ)
end

EstimationToolkit.measurement_noise_cov(m::RelayRangeRate) =
    @SMatrix [m.σ_range^2 0.0; 0.0 m.σ_rate^2]

function relay_state(t, phase)
    n = sqrt(MU / R_GEO^3)
    θ = phase + n * t
    return SVector(R_GEO * cos(θ), R_GEO * sin(θ)), SVector(-R_GEO * n * sin(θ), R_GEO * n * cos(θ))
end

"""True if the segment between the two points clears the Earth (plus 100 km of atmosphere)."""
function in_view(r_sat, r_relay)
    d = r_relay - r_sat
    s = clamp(-dot(r_sat, d) / dot(d, d), 0.0, 1.0)
    return norm(r_sat + s * d) > R_EARTH + 100.0
end

function measurement_model(t, r_sat; σ_range = 0.010, σ_rate = 1e-6)
    for phase in (0.0, π)
        r, v = relay_state(t, phase)
        in_view(r_sat, r) && return RelayRangeRate(r, v, σ_range, σ_rate)
    end
    error("no relay in view at t = $t")   # cannot happen for LEO with two relays 180° apart
end

# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

function simulate_truth(; dt = 10.0, duration = 4 * 5677.0, t_burn = 8000.0,
                        burn_duration = 300.0, a_burn = 1e-4)
    r0 = R_EARTH + 500.0
    x = SVector(r0, 0.0, 0.0, sqrt(MU / r0))
    times = collect(0.0:dt:duration)
    X = [x]
    for k in 2:length(times)
        # integrate with 1 s substeps so the burn starts and stops on time
        for s in 0:Int(dt) - 1
            t = times[k - 1] + s
            v = SVector(x[3], x[4])
            a = (t_burn <= t < t_burn + burn_duration) ? a_burn * v / norm(v) : SVector(0.0, 0.0)
            x = rk4(x, 1.0, a)
        end
        push!(X, x)
    end
    return times, X
end

function run_maneuver_example(; seed = 2026, dt = 10.0, verbose = true)
    rng = MersenneTwister(seed)
    t_burn, burn_duration = 8000.0, 300.0
    times, X = simulate_truth(; dt, t_burn, burn_duration)

    meas_models = [measurement_model(t, SVector(x[1], x[2])) for (t, x) in zip(times, X)]
    Y = [measure(m, x) + SVector(m.σ_range * randn(rng), m.σ_rate * randn(rng))
         for (m, x) in zip(meas_models, X)]

    # Process noise levels: ~1e-9 km/s² (1 µm/s²) unmodeled accel when coasting,
    # ~1e-4 km/s² (0.1 m/s²) when maneuvering.  q ≈ a² · dt.
    coast, burn = TwoBodyWNA(1e-17), TwoBodyWNA(1e-7)

    P0 = SMatrix{4, 4}(Diagonal([1.0, 1.0, 1e-6, 1e-6]))     # 1 km, 1 m/s (1σ)
    x0 = X[1] + cholesky(Matrix(P0)).L * randn(rng, 4)
    x0 = SVector{4}(x0)

    Π = [0.995 0.005;    # coast -> burn about once per 2000 s
         0.05  0.95]     # burns last ~200 s
    filters = (
        ekf_coast = ExtendedKalmanFilter(x0, P0),
        ekf_burn  = ExtendedKalmanFilter(x0, P0),
        imm  = InteractingMultipleModel([ExtendedKalmanFilter(x0, P0), ExtendedKalmanFilter(x0, P0)],
                                        Π, [0.99, 0.01]),
        gpb2 = GPB2Filter(ExtendedKalmanFilter(x0, P0), 2, Π, [0.99, 0.01]),
    )

    K = length(times)
    err = Dict(name => zeros(K) for name in keys(filters))
    sig = Dict(name => zeros(K) for name in keys(filters))
    p_burn = Dict(:imm => zeros(K), :gpb2 => zeros(K))
    u = zeros(0)

    record!(name, f, k) = begin
        x̂, P = state_estimate(f), covariance(f)
        err[name][k] = 1e3 * norm(SVector(x̂[1], x̂[2]) - SVector(X[k][1], X[k][2]))   # m
        sig[name][k] = 1e3 * sqrt(P[1, 1] + P[2, 2])
    end
    for name in keys(filters)
        record!(name, filters[name], 1)
    end

    for k in 2:K
        m, y = meas_models[k], Y[k]
        predict!(filters.ekf_coast, coast, u, dt); update!(filters.ekf_coast, m, y)
        predict!(filters.ekf_burn, burn, u, dt);   update!(filters.ekf_burn, m, y)
        step!(filters.imm,  [coast, burn], [m, m], u, y, dt)
        step!(filters.gpb2, [coast, burn], [m, m], u, y, dt)
        for name in keys(filters)
            record!(name, filters[name], k)
        end
        p_burn[:imm][k] = get_mode_probabilities(filters.imm)[2]
        p_burn[:gpb2][k] = get_mode_probabilities(filters.gpb2)[2]
    end

    windows = (
        before = times .< t_burn,
        burn   = t_burn .<= times .< t_burn + 1800.0,
        after  = times .>= t_burn + 1800.0,
    )
    rms(v) = sqrt(sum(abs2, v) / length(v))

    if verbose
        println("RMS position error [m]  (burn at t = $(Int(t_burn)) s, Δv = 30 m/s)")
        @printf("%-12s %10s %16s %10s\n", "filter", "before", "burn + 30 min", "after")
        for name in keys(filters)
            @printf("%-12s %10.2f %16.2f %10.2f\n", name,
                    rms(err[name][windows.before .& (times .> 1800)]),
                    rms(err[name][windows.burn]), rms(err[name][windows.after]))
        end
        detect(p) = (i = findfirst(>(0.5), p .* (times .>= t_burn)); i === nothing ? NaN : times[i] - t_burn)
        @printf("\nIMM  first P(burn) > 0.5: %.0f s after burn start\n", detect(p_burn[:imm]))
        @printf("GPB2 first P(burn) > 0.5: %.0f s after burn start\n", detect(p_burn[:gpb2]))
    end

    return (; times, X, err, sig, p_burn, t_burn, burn_duration)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_maneuver_example()
end
