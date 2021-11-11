using Pkg

Pkg.activate(".")

using DataFrames
using DataFramesMeta
using Distances
using CSV
using Chain
using NearestNeighbors
using StatsBase
using Pipe
using Transducers
using Revise

include("EncodingUtils.jl")
using .EncodingUtils

include("kNN.jl")
using .kNN


@doc """
    generate_search_columns(num_columns::Int64, column_name_templates::NTuple{N, String}) where N

Generates column names from `column_name_templates`.
"""
function generate_search_columns(num_columns::Int64, column_name_templates::NTuple{N, String}) where N
    colnames = Tuple{String, String}[]

    for i in 1:num_columns
        push!(colnames, ("$(column_name_templates[1])$i", "$(column_name_templates[2])$i"))
    end

    Iterators.flatten(colnames) |> collect

end


@doc """
   similarity_search(name::String, encoding_model::Tuple{Symbol, Symbol}, encoded_speech_acts::DataFrame, encoded_activities::DataFrame)

Computes the most similar keys for each speech act in `encoded_speech_acts` for a given `name`. \n
The encoding model needs to be given as a tuple.
"""
function similarity_search(name::String, encoding_model::Tuple{Symbol, Symbol}, encoded_speech_acts::DataFrame, encoded_activities::DataFrame, num_results::Int64)

    # Generating column names for result DataFrames
    columns = generate_search_columns(num_results, ("search_results_", "result_relevance_"))
    pushfirst!(columns, "original_text")

    search_results = []

    # Subsetting speech act data
    speaker_subset = @subset encoded_speech_acts begin
        :actor_name .== name
    end

    # Subsetting deputy activities
    activity_subset = @subset encoded_activities begin
        :name .== name
    end

    activity_data_dense = activity_subset[:, encoding_model[2]] |> collect

    # Iterating over every speech act row
    for row in eachrow(speaker_subset)
        sim_indices, sims = compute_nearest_neighbours(row[encoding_model[1]], activity_data_dense, 10)

        result_tuples = Tuple{String, Float32}[]

        for result_num in 1:min(num_results, length(activity_data_dense))
            search_result = activity_subset[sim_indices[result_num], :text]
            result_relevance = sims[result_num]
            push!(result_tuples, (search_result, result_relevance))
        end

        flat_results = Iterators.flatten(result_tuples) |> collect

        # The search sensitivity needs to be dampened
        # as some results might not be too relevant or at all.
        if flat_results[2] > 0.75

            pushfirst!(flat_results, row[:sentence_text])

            search_result_with_text = flat_results |> Tuple

            push!(search_results, search_result_with_text)
        end

    end

    search_results_df = DataFrame(search_results)

    # Shortcircuit function execution if no relevant results are retrieved
    if nrow(search_results_df) == 0
        return nothing
    end

    rename!(search_results_df, Symbol.(columns))

    relevances = mean.(eachcol(search_results_df[:, Cols(r"relevance")]))

    subset!(search_results_df, first(Symbol.(columns)) => ByRow(x -> !isnothing(x)))

    # Creating metadata for individual
    search_meta_data = Dict(
        :name => name,
        :relevances => relevances,
        :nrow_speech_acts => nrow(speaker_subset),
        :nrow_activities => nrow(activity_subset)
    )

    search_results_df, search_meta_data
end
