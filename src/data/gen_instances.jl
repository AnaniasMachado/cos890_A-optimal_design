using CSV
using DataFrames
using XLSX
using Statistics
using DelimitedFiles
using Printf

# ==================================================================
# Configuration
# ==================================================================

const OUTPUT_DIRECTORY = "data/aopt_matrices"

mkpath(OUTPUT_DIRECTORY)

# ==================================================================
# Dataset readers
# ==================================================================

function read_delimited_file(filename::String; names=nothing, delim=' ', missingstring="?")
    df = CSV.read(
        filename,
        DataFrame;
        delim=delim,
        ignorerepeated=true,
        header=false,
        missingstring=missingstring,
        stripwhitespace=true,
    )

    if names !== nothing
        ncol(df) == length(names) || error(
            "Expected $(length(names)) columns in $filename, but read $(ncol(df))."
        )

        rename!(df, names)
    end

    return df
end

function read_xlsx_first_sheet(filename::String)
    workbook = XLSX.readxlsx(filename)
    sheet_name = first(XLSX.sheetnames(workbook))
    return DataFrame(XLSX.gettable(workbook[sheet_name]))
end

function read_auto_mpg(filename::String)
    rows = NamedTuple[]

    for line in eachline(filename)
        line = strip(line)
        isempty(line) && continue

        fields = split(line, r"\s+"; limit=9)
        length(fields) == 9 || error("Unexpected Auto MPG row: $line")

        horsepower = fields[4] == "?" ? missing : parse(Float64, fields[4])

        push!(
            rows,
            (
                mpg=parse(Float64, fields[1]),
                cylinders=parse(Int, fields[2]),
                displacement=parse(Float64, fields[3]),
                horsepower=horsepower,
                weight=parse(Float64, fields[5]),
                acceleration=parse(Float64, fields[6]),
                model_year=parse(Int, fields[7]),
                origin=parse(Int, fields[8]),
                car_name=replace(fields[9], "\"" => ""),
            ),
        )
    end

    return DataFrame(rows; copycols=false)
end

# ==================================================================
# Data cleaning and construction of the raw AOPT matrix
#
# A has dimensions:
#
#     number of predictors × number of observations
#
# Each row of A is one predictor.
# Each column of A is one candidate observation.
# ==================================================================

function construct_raw_A(df::DataFrame, predictor_columns; dataset_name="dataset")
    predictor_df = select(df, predictor_columns)

    original_observation_count = nrow(predictor_df)

    # Remove observations containing missing predictor values.
    complete_row_mask = completecases(predictor_df)
    predictor_df = predictor_df[complete_row_mask, :]

    removed_missing = original_observation_count - nrow(predictor_df)

    # Convert all selected predictors to Float64.
    X = Matrix{Float64}(predictor_df)

    # Remove observations containing NaN or Inf.
    finite_row_mask = [all(isfinite, view(X, i, :)) for i in axes(X, 1)]
    X = X[finite_row_mask, :]

    removed_nonfinite = count(!, finite_row_mask)

    isempty(X) && error("$dataset_name has no usable observations after cleaning.")

    # Detect predictors with zero or numerically negligible range.
    retained_predictor_mask = trues(size(X, 2))
    removed_predictors = String[]

    for j in axes(X, 2)
        minimum_value = minimum(view(X, :, j))
        maximum_value = maximum(view(X, :, j))
        predictor_range = maximum_value - minimum_value
        scale = max(abs(minimum_value), abs(maximum_value), 1.0)

        if predictor_range <= eps(Float64) * scale
            retained_predictor_mask[j] = false
            push!(removed_predictors, string(predictor_columns[j]))
        end
    end

    X = X[:, retained_predictor_mask]
    retained_predictors = string.(predictor_columns[retained_predictor_mask])

    size(X, 2) > 0 || error("$dataset_name has no nonconstant predictors.")

    # In the AOPT formulation, observations are columns.
    A_raw = Matrix(transpose(X))

    println()
    println(dataset_name)
    println("-"^length(dataset_name))
    @printf("original observations       : %d\n", original_observation_count)
    @printf("removed missing observations : %d\n", removed_missing)
    @printf("removed nonfinite observations: %d\n", removed_nonfinite)
    @printf("retained observations        : %d\n", size(A_raw, 2))
    @printf("retained predictors          : %d\n", size(A_raw, 1))

    if !isempty(removed_predictors)
        println("removed constant predictors   : ", join(removed_predictors, ", "))
    end

    return A_raw, retained_predictors
end

# ==================================================================
# Transformations
#
# Transformations are applied row-wise because predictors are rows
# of A and observations are columns.
# ==================================================================

function standardize_A(A_raw::Matrix{Float64}; dataset_name="dataset")
    A_standardized = similar(A_raw)

    for i in axes(A_raw, 1)
        values = view(A_raw, i, :)
        μ = mean(values)
        σ = std(values)

        if !isfinite(σ) || σ <= eps(Float64)
            error(
                "Cannot standardize predictor row $i in $dataset_name " *
                "because its standard deviation is zero or nonfinite."
            )
        end

        A_standardized[i, :] .= (values .- μ) ./ σ
    end

    return A_standardized
end

function normalize_A(A_raw::Matrix{Float64}; dataset_name="dataset")
    A_normalized = similar(A_raw)

    for i in axes(A_raw, 1)
        values = view(A_raw, i, :)
        minimum_value = minimum(values)
        maximum_value = maximum(values)
        predictor_range = maximum_value - minimum_value

        if !isfinite(predictor_range) || predictor_range <= eps(Float64)
            error(
                "Cannot normalize predictor row $i in $dataset_name " *
                "because its range is zero or nonfinite."
            )
        end

        A_normalized[i, :] .= (values .- minimum_value) ./ predictor_range
    end

    return A_normalized
end

# ==================================================================
# Saving
# ==================================================================

function save_matrix(filename::String, A::Matrix{Float64})
    writedlm(filename, A, ',')

    @printf(
        "saved %-55s size = %d × %d\n",
        filename,
        size(A, 1),
        size(A, 2),
    )
end

function save_predictor_names(filename::String, predictor_names::Vector{String})
    open(filename, "w") do file
        for predictor in predictor_names
            println(file, predictor)
        end
    end

    println("saved ", filename)
end

function process_and_save_dataset(dataset_name::String, file_prefix::String, df::DataFrame, predictor_columns)
    A_raw, retained_predictors = construct_raw_A(
        df,
        predictor_columns;
        dataset_name=dataset_name,
    )

    A_standardized = standardize_A(
        A_raw;
        dataset_name=dataset_name,
    )

    A_normalized = normalize_A(
        A_raw;
        dataset_name=dataset_name,
    )

    raw_filename = OUTPUT_DIRECTORY * "/A_" * file_prefix * "_raw.csv"
    standardized_filename = OUTPUT_DIRECTORY * "/A_" * file_prefix * "_standardized.csv"
    normalized_filename = OUTPUT_DIRECTORY * "/A_" * file_prefix * "_normalized.csv"
    predictors_filename = OUTPUT_DIRECTORY * "/A_" * file_prefix * "_predictors.txt"

    save_matrix(raw_filename, A_raw)
    save_matrix(standardized_filename, A_standardized)
    save_matrix(normalized_filename, A_normalized)
    save_predictor_names(predictors_filename, retained_predictors)

    return A_raw, A_standardized, A_normalized
end

# ==================================================================
# Airfoil Self-Noise
#
# Response:
#     sound_pressure_level
# ==================================================================

airfoil = read_delimited_file(
    "data/airfoil_self_noise.dat";
    delim='\t',
    names=[
        "frequency",
        "attack_angle",
        "chord_length",
        "free_stream_velocity",
        "displacement_thickness",
        "sound_pressure_level",
    ],
)

airfoil_predictors = [
    "frequency",
    "attack_angle",
    "chord_length",
    "free_stream_velocity",
    "displacement_thickness",
]

A_airfoil_raw, A_airfoil_standardized, A_airfoil_normalized = process_and_save_dataset(
    "Airfoil Self-Noise",
    "airfoil",
    airfoil,
    airfoil_predictors,
)

# ==================================================================
# Auto MPG
#
# Response:
#     mpg
#
# Excluded:
#     car_name
#
# Rows containing missing horsepower values are removed.
# ==================================================================

auto_mpg = read_auto_mpg("data/auto-mpg.data")

auto_mpg_predictors = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
]

A_auto_mpg_raw, A_auto_mpg_standardized, A_auto_mpg_normalized = process_and_save_dataset(
    "Auto MPG",
    "auto_mpg",
    auto_mpg,
    auto_mpg_predictors,
)

# ==================================================================
# Concrete Compressive Strength
#
# The first eight columns are predictors.
# The final column is the response.
# ==================================================================

concrete = read_xlsx_first_sheet("data/Concrete_Data.xlsx")
concrete_predictors = names(concrete)[1:8]

A_concrete_raw, A_concrete_standardized, A_concrete_normalized = process_and_save_dataset(
    "Concrete Compressive Strength",
    "concrete",
    concrete,
    concrete_predictors,
)

# ==================================================================
# Energy Efficiency
#
# Predictors:
#     X1, ..., X8
#
# Responses:
#     Y1 and Y2
#
# Both responses share the same predictor matrix.
# ==================================================================

energy = read_xlsx_first_sheet("data/ENB2012_data.xlsx")

energy_predictors = [
    "X1",
    "X2",
    "X3",
    "X4",
    "X5",
    "X6",
    "X7",
    "X8",
]

A_energy_raw, A_energy_standardized, A_energy_normalized = process_and_save_dataset(
    "Energy Efficiency",
    "energy",
    energy,
    energy_predictors,
)

# ==================================================================
# Forest Fires
#
# Response:
#     area
#
# The categorical month and day variables are excluded.
# ==================================================================

forest_fires = CSV.read(
    "data/forestfires.csv",
    DataFrame;
    missingstring=["?", "NA", ""],
)

forest_fires_predictors = [
    "X",
    "Y",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "temp",
    "RH",
    "wind",
    "rain",
]

A_forest_fires_raw, A_forest_fires_standardized, A_forest_fires_normalized = process_and_save_dataset(
    "Forest Fires",
    "forest_fires",
    forest_fires,
    forest_fires_predictors,
)

# ==================================================================
# Wine Quality — Red
#
# Response:
#     quality
# ==================================================================

wine_red = CSV.read(
    "data/winequality-red.csv",
    DataFrame;
    delim=';',
    missingstring=["?", "NA", ""],
)

wine_red_predictors = names(wine_red)[1:11]

A_wine_red_raw, A_wine_red_standardized, A_wine_red_normalized = process_and_save_dataset(
    "Wine Quality — Red",
    "wine_red",
    wine_red,
    wine_red_predictors,
)

# ==================================================================
# Wine Quality — White
#
# Response:
#     quality
# ==================================================================

wine_white = CSV.read(
    "data/winequality-white.csv",
    DataFrame;
    delim=';',
    missingstring=["?", "NA", ""],
)

wine_white_predictors = names(wine_white)[1:11]

A_wine_white_raw, A_wine_white_standardized, A_wine_white_normalized = process_and_save_dataset(
    "Wine Quality — White",
    "wine_white",
    wine_white,
    wine_white_predictors,
)

# ==================================================================
# Yacht Hydrodynamics
#
# Response:
#     residuary_resistance
# ==================================================================

yacht = read_delimited_file(
    "data/yacht_hydrodynamics.data";
    delim=' ',
    names=[
        "longitudinal_position",
        "prismatic_coefficient",
        "length_displacement_ratio",
        "beam_draught_ratio",
        "length_beam_ratio",
        "froude_number",
        "residuary_resistance",
    ],
)

yacht_predictors = [
    "longitudinal_position",
    "prismatic_coefficient",
    "length_displacement_ratio",
    "beam_draught_ratio",
    "length_beam_ratio",
    "froude_number",
]

A_yacht_raw, A_yacht_standardized, A_yacht_normalized = process_and_save_dataset(
    "Yacht Hydrodynamics",
    "yacht",
    yacht,
    yacht_predictors,
)

println()
println("All AOPT matrices were saved in: ", OUTPUT_DIRECTORY)