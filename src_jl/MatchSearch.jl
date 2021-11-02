using DataFrames
using DataFramesMeta
using CSV
using Chain
using NearestNeighbors
using StatsBase
using Pipe
using ProgressMeter
using Revise

@doc """
    parse_encoding(encoding::String)

Given a string `encoding` of an array, returns a float array.
"""
function parse_encoding(encoding::String)
    @chain encoding begin
        replace(_, r"[\[\]\)\(\n]" => s"") |> strip
        split(_, " ")
        filter(!isempty, _) .|> String
        parse.(Float32, _)
    end
end


@doc """
    parse_phrases(phrase::String)

Given a string `phrase` with an array of phrase strings, returns a string array.
"""
function parse_phrases(phrase::String)
    @chain phrase begin
        replace(_, r"[\]\[\n\'\)\(]" => s"") |> strip
        split(_, ", ")
        filter(!isempty, _) .|> String
    end
end


@doc """
    compute_nearest_neighbours(comparison_point::Vector{Float32}, data::Vector{Vector{Float32}}, num_neighbors::Int64)::Tuple{Int64, Float32}

Computes the `num_neighbors` from `data`. The data is supposed to be in the \n
`encoding_column`. Returns the indices and distances.
"""
function compute_nearest_neighbours(comparison_point::Vector{Float32}, data::Vector{Vector{Float32}}, num_neighbors::Int64)::Tuple{Vector{Int64}, Vector{Float32}}

    # Creating data matrices
    data_matrix = hcat(data...)

    # Instantiating the splitting tree
    balltree = BallTree(data_matrix, Minkowski(3); reorder = false)

    # Computing indices and distances
    # Catching the case when there are more requested neighbors than data points
    if size(data_matrix)[2] >= num_neighbors
        idxs, dists = knn(balltree, comparison_point, Int(floor(size(data_matrix)[2]/2)), true)
    else
        idxs, dists = knn(balltree, comparison_point, num_neighbors, true)
    end

    idxs, dists
end


@doc """
   similarity_search(name::String, encoding_model::Tuple{Symbol, Symbol}, encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::DataFrame

Computes the most similar keys for each speech act in `encoded_speech_acts` for a given `name`. \n
The encoding model needs to be given as a tuple.
"""
function similarity_search(name::String, encoding_model::Tuple{Symbol, Symbol}, encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::DataFrame

    # Instantiating summary stats dataframe
    statistics = DataFrame(
        sim_searches = Vector{Int64}[],
        average_dist = Float32[]
    )

    # Subsetting speech act data
    speaker_subset = @subset encoded_speech_acts begin
        :actor_name .== name
    end

    # Subsetting deputy activities
    activity_subset = @subset encoded_activities begin
        :name .== name
    end

    activity_data_dense = activity_subset[:, encoding_model[2]] |> collect

    for row in eachrow(speaker_subset)
        sim_indices, sim_dists = compute_nearest_neighbours(row[encoding_model[1]], activity_data_dense, 10)
        push!(statistics, [sim_indices, mean(sim_dists)])
    end

    statistics
end

sample_names(names::Vector{String}, number_of_names::Int64) = sample(names, number_of_names)

@doc """
    validate_data_integrity(encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::Vector{String}

Returns names that are in both `encoded_speech_acts` and `encoded_activities`.
"""
function validate_data_integrity(encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::Vector{String}
    speech_act_names = encoded_speech_acts[:, :actor_name]
    activity_names = encoded_activities[:, :name]

    intersect(speech_act_names, activity_names)
end


@doc """
    validate_search(name::String, encoding_model::Tuple{Symbol, Symbol}, encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::Tuple{Float32, Float32}

Computes two evaluation metrics for a `name` query search.
"""
function validate_search(name::String, encoding_model::Tuple{Symbol, Symbol}, encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::Tuple{Float32, Float32}
    statistics = similarity_search(name, encoding_model, encoded_speech_acts, encoded_activities)

    speaker_subset = @subset encoded_speech_acts begin
        :actor_name .== name
    end

    activity_subset = @subset encoded_activities begin
        :name .== name
    end

    all_activity_phrases = []

    for activity_statistic in eachrow(statistics)
        phrase_indices = activity_statistic[:sim_searches]

        phrases = Vector{String}[]

        foreach(phrase_indices) do p
            activity_phrases = activity_subset[p, :activity_phrases]
            push!(phrases, activity_phrases)
        end

        push!(all_activity_phrases, reduce(vcat, phrases))
    end

    average_keyword_matches = @pipe intersect.(speaker_subset[:, :speech_act_phrases], all_activity_phrases) |>
                            length.(_) |>
                            mean

    aggregate_distance = statistics[:, :average_dist] |> mean

    average_keyword_matches, aggregate_distance
end


@doc """
    get_performance_metrics(names::Vector{String}, encoding_model::Tuple{Symbol, Symbol}, encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::Tuple{Float32, Float32}

Iterative version of `validate_search`. Iterates over `names` and generates aggregate performance statistics.
"""
function get_performance_metrics(names::Vector{String}, encoding_model::Tuple{Symbol, Symbol}, encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::Tuple{Float32, Float32}

    stats = Tuple{Float32, Float32}[]

    n = length(names)
    p = Progress(n, 1)

    for name in names
        push!(stats, validate_search(name, encoding_model, encoded_speech_acts, encoded_activities))
        next!(p)
    end

    mean(getindex.(stats, 1)), mean(getindex.(stats, 2))
end


validate_data_integrity(encoded_speech_acts, encoded_activities)

encoded_activities = DataFrame(CSV.File("./data/encoded_datasets/activities_encoded.csv"))

transform!(encoded_activities, :encoded_activities_ml => ByRow(x -> parse_encoding(x)) => :encoded_activities_ml)
transform!(encoded_activities, :encoded_activities_pt => ByRow(x -> parse_encoding(x)) => :encoded_activities_pt)
transform!(encoded_activities, :activity_phrases => ByRow(x -> parse_phrases(x)) => :activity_phrases)

encoded_speech_acts = DataFrame(CSV.File("./data/encoded_datasets/speech_acts_encoded.csv"))

transform!(encoded_speech_acts, :encoded_speech_acts_ml => ByRow(x -> parse_encoding(x)) => :encoded_speech_acts_ml)
transform!(encoded_speech_acts, :encoded_speech_acts_pt => ByRow(x -> parse_encoding(x)) => :encoded_speech_acts_pt)
transform!(encoded_speech_acts, :speech_act_phrases => ByRow(x -> parse_phrases(x)) => :speech_act_phrases)

politician_names = @pipe validate_data_integrity(encoded_speech_acts, encoded_activities) |>
                Set .|>
                String |>
                collect |>
                sample_names(_, 100)

result_ml = get_performance_metrics(politician_names, (:encoded_speech_acts_ml, :encoded_activities_ml), encoded_speech_acts, encoded_activities)

result_pt = get_performance_metrics(politician_names, (:encoded_speech_acts_pt, :encoded_activities_pt), encoded_speech_acts, encoded_activities)
