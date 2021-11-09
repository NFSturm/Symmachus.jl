using DataFrames
using DataFramesMeta
using Distances
using CSV
using Chain
using NearestNeighbors
using StatsBase
using Pipe
using Revise

include("EncodingUtils.jl")
using .EncodingUtils

include("kNN.jl")
using .kNN

@doc """
   similarity_search(name::String, encoding_model::Tuple{Symbol, Symbol}, encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::DataFrame

Computes the most similar keys for each speech act in `encoded_speech_acts` for a given `name`. \n
The encoding model needs to be given as a tuple.
"""
function similarity_search(name::String, encoding_model::Tuple{Symbol, Symbol}, encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::DataFrame

    # Instantiating summary stats dataframe
    search_results = DataFrame(
        best_search_result = String[],
        best_result_relevance = Float32,
        second_best_result = String[],
        second_best_relevance = Float32[]
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
        sim_indices, sims = compute_nearest_neighbours(row[encoding_model[1]], activity_data_dense, 10)

        # Retrieving best results and relevance metric
        best_search_result = activity_subset[sim_indices[1], :text]
        best_result_relevance = sims[1]

        second_best_result = activity_subset[sim_indices[2], :text]
        second_best_relevance = sims[2]

        push!(search_results,
                [best_search_result,
                best_result_relevance,
                second_best_result,
                second_best_relevance])
    end

    search_results
end
