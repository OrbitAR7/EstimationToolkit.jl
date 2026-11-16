# Figures for examples/orbit_uncertainty.jl.
# Needs Plots.jl in the active environment (it is not a package dependency):
#   julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path="."); Pkg.add(["Plots", "StaticArrays"])' \
#         -L examples/plot_orbit_uncertainty.jl

using Plots
include(joinpath(@__DIR__, "orbit_uncertainty.jl"))

outdir = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(outdir)

# ---------------------------------------------------------------------------
# 1. The cloud after the gap, in the orbit plane
# ---------------------------------------------------------------------------

p = run_propagation()
(; x0, dyn) = p.sc

# Along-track / radial axes of the reference orbit at the end of the gap
x_ref = propagate(dyn, x0, nothing, p.times[end])
r_ref, v_ref = x_ref[SOneTo(3)], SVector(x_ref[4], x_ref[5], x_ref[6])
r̂ = r_ref / norm(r_ref)
ĥ = cross(r_ref, v_ref); ĥ /= norm(ĥ)
t̂ = cross(ĥ, r̂)
J = vcat(t̂', r̂')                                   # 3-D position -> (along-track, radial)

project(x) = J * (x[SOneTo(3)] - r_ref)
project_cov(P) = J * P[SOneTo(3), SOneTo(3)] * J'

"""Points on the k-sigma ellipse of a 2-D Gaussian."""
function ellipse(μ, P; k = 3.0, n = 200)
    E = eigen(Symmetric(Matrix(P)))
    A = E.vectors * Diagonal(sqrt.(max.(E.values, 0.0)))
    pts = [μ + k * A * [cos(θ), sin(θ)] for θ in range(0, 2π; length = n)]
    return first.(pts), last.(pts)
end

cloud = project.(p.samples[1:4:end])
cx, cy = first.(cloud), last.(cloud)
xl = (minimum(cx) - 40, maximum(cx) + 40)
yl = (minimum(cy) - 3, maximum(cy) + 9)

function base(title)
    plt = scatter(cx, cy; ms = 1.2, msw = 0, color = :gray60, alpha = 0.5, label = "Monte Carlo",
                  xlabel = "along-track [km]", ylabel = "radial [km]", title, titlefontsize = 10,
                  xlims = xl, ylims = yl, legend = :topright, legendfontsize = 7)
    return plt
end

pa = base("One Gaussian")
for (f, name, c) in ((p.ekf, "EKF 3σ", :firebrick), (p.ukf, "UKF 3σ", :royalblue))
    ex, ey = ellipse(project(state_estimate(f)), project_cov(covariance(f)))
    plot!(pa, ex, ey; lw = 1.8, color = c, label = name)
end

mix = p.mixtures[2]
pb = base("Adaptive mixture, $(num_components(mix)) components")
for (j, (μ, P)) in enumerate(zip(mix.means, mix.covs))
    ex, ey = ellipse(project(μ), project_cov(P); k = 2.0)
    plot!(pb, ex, ey; lw = 1.0, color = :seagreen, alpha = 0.45 + 0.55 * mix.weights[j] / maximum(mix.weights),
          label = j == 1 ? "components 2σ" : "")
end

fig1 = plot(pa, pb; layout = (1, 2), size = (1100, 430), left_margin = 6Plots.mm, bottom_margin = 6Plots.mm,
            plot_title = "Position uncertainty after 2 revolutions without tracking (2 km, 2 m/s initially)",
            plot_titlefontsize = 11)
savefig(fig1, joinpath(outdir, "orbit_uncertainty_cloud.png"))

# ---------------------------------------------------------------------------
# 2. KL divergence during the gap
# ---------------------------------------------------------------------------

revs = p.times ./ p.sc.period
colors = [:firebrick, :royalblue, :darkseagreen3, :seagreen, :darkgreen]
fig2 = plot(; yscale = :log10, xlabel = "revolutions since last measurement",
            ylabel = "KL divergence from truth [nats]", ylims = (1e-2, 1e4), legend = :topleft,
            title = "How far each filter's density is from the true one", titlefontsize = 10,
            size = (900, 420), left_margin = 6Plots.mm, bottom_margin = 5Plots.mm)
for (i, l) in enumerate(p.labels)
    plot!(fig2, revs[2:end], max.(p.kl[i][2:end], 1e-2); lw = 1.8, color = colors[i], label = l,
          ls = i <= 2 ? :solid : :dash, marker = :circle, ms = 2.5, msw = 0)
end
savefig(fig2, joinpath(outdir, "orbit_uncertainty_kl.png"))

# ---------------------------------------------------------------------------
# 3. Tracking pass
# ---------------------------------------------------------------------------

t = run_tracking()
names = (ekf = "EKF", ukf = "UKF", mixture = "adaptive mixture")
cols = (ekf = :firebrick, ukf = :royalblue, mixture = :seagreen)
med(A) = [median(A[:, k]) for k in axes(A, 2)]

pc = plot(; yscale = :log10, xlabel = "time into the pass [s]", ylabel = "median NEES",
          title = "Consistency (200 trials)", titlefontsize = 10, legend = :right, ylims = (1, 1e6))
hline!(pc, [5.35]; color = :black, ls = :dot, label = "χ²(6) median")
hline!(pc, [t.χ2_99]; color = :black, ls = :dash, label = "χ²(6) 99 %")
pd = plot(; yscale = :log10, xlabel = "time into the pass [s]", ylabel = "median position error [km]",
          title = "Accuracy", titlefontsize = 10, legend = :topright, ylims = (0.1, 100))
for n in t.names
    plot!(pc, t.times, med(t.nees[n]); lw = 1.8, color = cols[n], label = names[n])
    plot!(pd, t.times, med(t.pos_err[n]); lw = 1.8, color = cols[n], label = names[n])
end
fig3 = plot(pc, pd; layout = (1, 2), size = (1100, 400), left_margin = 6Plots.mm, bottom_margin = 6Plots.mm,
            plot_title = "Radar pass (range 10 m, range-rate 1 cm/s) after the 2-revolution gap",
            plot_titlefontsize = 11)
savefig(fig3, joinpath(outdir, "orbit_uncertainty_tracking.png"))

println("Figures written to ", normpath(outdir))
