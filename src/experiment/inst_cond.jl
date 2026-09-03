using LinearAlgebra
using Printf
using DelimitedFiles

const INSTANCE_NAMES = [
    "airfoil",
    "auto_mpg",
    "concrete",
    "energy",
    "forest_fires",
    "wine_red",
    "wine_white",
    "yacht",
]

const PREPROCESSINGS = [
    "raw",
    "normalized",
    "standardized",
]

const DATA_FOLDER = "data/aopt_matrices"


function read_instance(instance_name::String, preprocessing::String)
    filename = "A_$(instance_name)_$(preprocessing).csv"
    path = joinpath(DATA_FOLDER, filename)

    isfile(path) || error("Instance file not found: $path")

    return Matrix{Float64}(
        readdlm(
            path,
            ',',
        ),
    )
end


println("Conditioning check for AOPT instances")
println("-------------------------------------")

for instance_name in INSTANCE_NAMES
    for preprocessing in PREPROCESSINGS
        A = read_instance(
            instance_name,
            preprocessing,
        )

        m, n = size(A)

        r = rank(A)

        G = Symmetric(A * A')
        λ = eigvals(G)

        λmin = minimum(λ)
        λmax = maximum(λ)

        κ = cond(Matrix(G))

        full_rank = r == m

        @printf(
            "%-15s %-12s  m=%2d  n=%5d  rank=%2d  full_rank=%5s  λmin=% .3e  λmax=% .3e  cond=% .3e\n",
            instance_name,
            preprocessing,
            m,
            n,
            r,
            string(full_rank),
            λmin,
            λmax,
            κ,
        )
    end
end