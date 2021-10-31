using DataFrames
using DataFramesMeta
using CSV
using Chain
using NearestNeighbors
using StatsBase
using Revise

@doc """
    parse_encoding(encoding::String)

Given a string `encoding` of an array, returns a float array.
"""
function parse_encoding(encoding::String)
    @chain encoding begin
        replace(_, r"[\]\[\n]\)\(" => s"") |> strip
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
    idxs, dists = knn(balltree, comparison_point, num_neighbors, true)

    idxs, dists
end

encoded_activities = DataFrame(CSV.File("./data/encoded_datasets/activities_encoded.csv"))

transform!(encoded_activities, :encoded_activities_ml => ByRow(x -> parse_encoding(x)) => :encoded_activities_ml)
transform!(encoded_activities, :encoded_activities_pt => ByRow(x -> parse_encoding(x)) => :encoded_activities_pt)
transform!(encoded_activities, :activity_phrases => ByRow(x -> parse_phrases(x)) => :activity_phrases)

encoded_speech_acts = DataFrame(CSV.File("./data/encoded_datasets/speech_acts_encoded.csv"))

transform!(encoded_speech_acts, :encoded_speech_acts_ml => ByRow(x -> parse_encoding(x)) => :encoded_speech_acts_ml)
transform!(encoded_speech_acts, :encoded_speech_act => ByRow(x -> parse_encoding(x)) => :encoded_speech_act)
transform!(encoded_speech_acts, :speech_act_phrases => ByRow(x -> parse_phrases(x)) => :speech_act_phrases)

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
