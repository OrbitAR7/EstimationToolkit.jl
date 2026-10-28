using Documenter
using EstimationToolkit

makedocs(
    sitename = "EstimationToolkit.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://orbitar7.github.io/EstimationToolkit.jl",
        assets = String[],
    ),
    modules = [EstimationToolkit],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Filter Catalog" => [
            "Linear Filters" => "filters/linear.md",
            "Extended Kalman Family" => "filters/extended.md",
            "Sigma-Point Filters" => "filters/sigma_point.md",
            "Particle Filters" => "filters/particle.md",
            "Ensemble Methods" => "filters/ensemble.md",
            "Multiple Model" => "filters/multiple_model.md",
            "Grid-Based" => "filters/grid.md",
        ],
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/OrbitAR7/EstimationToolkit.jl.git",
    devbranch = "main",
)
