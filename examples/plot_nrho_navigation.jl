# Figures for examples/nrho_navigation.jl.
# Needs Plots.jl and OrbitalMechanics.jl in the active environment:
#   julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path=".");
#             Pkg.add(url="https://github.com/OrbitAR7/orbital-mechanics-jl");
#             Pkg.add(["Plots", "StaticArrays"])' -L examples/plot_nrho_navigation.jl

using Plots
include(joinpath(@__DIR__, "nrho_navigation.jl"))

outdir = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(outdir)

# ---------------------------------------------------------------------------
# 1. Coast: KL divergence, with the distance from the Moon for orientation
# ---------------------------------------------------------------------------

c = run_coast()
colors = [:firebrick, :royalblue, :darkseagreen3, :darkgreen]

p_top = plot(c.days, c.moon_km ./ 1e3; color = :gray30, lw = 1.5, legend = false, ylabel = "distance from\nthe Moon [10³ km]",
             title = "Three revolutions of the 9:2 NRHO without tracking (1 km, 1 cm/s initially)",
             titlefontsize = 10, xformatter = _ -> "", ylims = (0, 80), bottom_margin = -2Plots.mm)
vline!(p_top, c.perilunes; color = :gray60, ls = :dot)

floor_value = 1e-2
p_kl = plot(; yscale = :log10, xlabel = "days since apolune", ylabel = "KL divergence from truth [nats]",
            ylims = (floor_value, 1e8), legend = :topleft, legendfontsize = 7,
            yticks = 10.0 .^ (-2:2:8))
vline!(p_kl, c.perilunes; color = :gray60, ls = :dot, label = "perilune")
for (i, l) in enumerate(c.labels)
    plot!(p_kl, c.days, max.(c.kl[i], floor_value); lw = i == 1 ? 1.6 : 1.8, color = colors[i], label = l,
          ls = i <= 2 ? :solid : :dash)
end

fig1 = plot(p_top, p_kl; layout = grid(2, 1, heights = [0.25, 0.75]), size = (1000, 620),
            left_margin = 8Plots.mm, bottom_margin = 5Plots.mm)
savefig(fig1, joinpath(outdir, "nrho_coast_kl.png"))

# ---------------------------------------------------------------------------
# 2. Tracking with an outage
# ---------------------------------------------------------------------------

t = run_tracking()
names = (ekf = "EKF", ukf = "UKF", mixture = "adaptive mixture")
cols = (ekf = :firebrick, ukf = :royalblue, mixture = :seagreen)
med(A) = [median(A[:, k]) for k in axes(A, 2)]

function shade!(plt)
    vspan!(plt, collect(t.outage_days); color = :orange, alpha = 0.18, label = "station down")
    vline!(plt, t.perilunes; color = :gray60, ls = :dot, label = "perilune")
end

p_err = plot(; yscale = :log10, ylabel = "median position error [km]", ylims = (0.01, 30),
             title = "Tracking from one ground station, 100 trials", titlefontsize = 10,
             legend = :topright, legendfontsize = 7, xformatter = _ -> "", bottom_margin = -2Plots.mm)
shade!(p_err)
p_nees = plot(; yscale = :log10, xlabel = "days since apolune", ylabel = "median NEES", ylims = (2, 60),
              legend = :topleft, legendfontsize = 7, yticks = ([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"]))
shade!(p_nees)
hline!(p_nees, [5.35]; color = :black, ls = :dot, label = "χ²(6) median")
hline!(p_nees, [t.χ2_99]; color = :black, ls = :dash, label = "χ²(6) 99 %")
for n in t.names
    plot!(p_err, t.days, med(t.pos_err[n]); lw = 1.6, color = cols[n], label = names[n])
    plot!(p_nees, t.days, med(t.nees[n]); lw = 1.6, color = cols[n], label = names[n])
end

fig2 = plot(p_err, p_nees; layout = (2, 1), size = (1000, 640), left_margin = 8Plots.mm, bottom_margin = 5Plots.mm)
savefig(fig2, joinpath(outdir, "nrho_tracking.png"))

println("Figures written to ", normpath(outdir))
