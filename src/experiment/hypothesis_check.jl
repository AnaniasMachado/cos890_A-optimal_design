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

const K_OVER_N_VALUES = [0.025, 0.05, 0.10]

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


println("Checking k > m for all instances")
println("--------------------------------")

all_valid = true

for instance_name in INSTANCE_NAMES
    for preprocessing in PREPROCESSINGS
        A = read_instance(
            instance_name,
            preprocessing,
        )

        m, n = size(A)

        for ratio in K_OVER_N_VALUES
            k = ceil(Int, ratio * n)
            valid = k > m

            @printf(
                "%-15s %-12s  m=%3d  n=%5d  k/n=%5.3f  k=%4d  valid=%s\n",
                instance_name,
                preprocessing,
                m,
                n,
                ratio,
                k,
                string(valid),
            )

            if !valid
                global all_valid = false
            end
        end
    end
end

println()
println("--------------------------------")

if all_valid
    println("All ratios satisfy k > m for every instance.")
else
    println("At least one instance has k <= m.")
end