# Figures for examples/orbit_maneuver_imm.jl.
# Needs Plots.jl in the active environment (it is not a package dependency):
#   julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path="."); Pkg.add("Plots")' \
#         -L examples/plot_orbit_maneuver.jl

using Plots
include(joinpath(@__DIR__, "orbit_maneuver_imm.jl"))

r = run_maneuver_example()
hours = r.times ./ 3600
tb, te = r.t_burn / 3600, (r.t_burn + r.burn_duration) / 3600
outdir = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(outdir)

labels = (ekf_coast = "EKF (ballistic Q)", ekf_burn = "EKF (maneuver Q)", imm = "IMM", gpb2 = "GPB2")
p1 = plot(yscale = :log10, xlabel = "time [h]", ylabel = "position error [m]",
          title = "LEO OD through a 30 m/s along-track burn", legend = :topright,
          size = (900, 420), ylims = (0.1, 1e5), left_margin = 6Plots.mm, bottom_margin = 5Plots.mm)
vspan!(p1, [tb, te], color = :orange, alpha = 0.25, label = "burn")
for name in (:ekf_coast, :ekf_burn, :imm, :gpb2)
    plot!(p1, hours[2:end], max.(r.err[name][2:end], 0.1), label = labels[name], lw = 1.2)
end
savefig(p1, joinpath(outdir, "maneuver_position_error.png"))

zoom = (r.t_burn - 1200 .<= r.times .<= r.t_burn + 2400)
p2 = plot(xlabel = "time from burn start [s]", ylabel = "P(maneuver mode)",
          title = "Mode probability", legend = :right, size = (900, 320), ylims = (-0.02, 1.02), left_margin = 6Plots.mm, bottom_margin = 6Plots.mm)
vspan!(p2, [0, r.burn_duration], color = :orange, alpha = 0.25, label = "burn")
plot!(p2, r.times[zoom] .- r.t_burn, r.p_burn[:imm][zoom], label = "IMM", lw = 1.5)
plot!(p2, r.times[zoom] .- r.t_burn, r.p_burn[:gpb2][zoom], label = "GPB2", lw = 1.5, ls = :dash)
savefig(p2, joinpath(outdir, "maneuver_mode_probability.png"))

println("Figures written to ", normpath(outdir))
