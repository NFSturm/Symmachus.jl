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
using Serialization
using ProgressMeter

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

    if ncol(search_results_df) == length(columns)
        rename!(search_results_df, Symbol.(columns))
    else
        number_of_columns = ncol(search_results_df)
        column_subset = columns[1:number_of_columns]
        rename!(search_results_df, Symbol.(column_subset))
    end


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

@doc """
    iterate_similarity_search(names::Vector{String}, encoding_model::Tuple{Symbol, Symbol}, encoded_speech_acts::DataFrame, encoded_activities::DataFrame, num_results::Int64)

Iterates of `names` and performs the similarity search.
"""
function iterate_similarity_search(names::Vector{String}, encoding_model::Tuple{Symbol, Symbol}, encoded_speech_acts::DataFrame, encoded_activities::DataFrame, num_results::Int64)

    result_container = []

    n = length(names)
    p = Progress(n, 1)

    for name in names
        res = similarity_search(name, encoding_model, encoded_speech_acts, encoded_activities, 3)
        push!(result_container, res)
        next!(p)
    end

    result_container
end

@doc """
    validate_name_integrity(encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::Vector{String}

Returns names that are in both `encoded_speech_acts` and `encoded_activities`.
"""
function validate_name_integrity(encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::Vector{String}
    speech_act_names = encoded_speech_acts[:, :actor_name]
    activity_names = encoded_activities[:, :name]

    intersect(speech_act_names, activity_names)
end

#****************** LOADING DATA ******************

@info "Loading speech acts…"

encoded_speech_acts = DataFrame(CSV.File("./data/encoded_datasets/speech_acts_encoded.csv"))

transform!(encoded_speech_acts, :encoded_speech_acts_nm => ByRow(x -> parse_encoding(x)) => :encoded_speech_acts_nm)
transform!(encoded_speech_acts, :actor_name => ByRow(x -> lowercase(x)) => :actor_name)

@info "Loading activities…"

encoded_activities = DataFrame(CSV.File("./data/encoded_datasets/activities_encoded.csv"))

transform!(encoded_activities, :encoded_activities_nm => ByRow(x -> parse_encoding(x)) => :encoded_activities_nm)
transform!(encoded_activities, :name => ByRow(x -> lowercase(x)) => :name)

politician_names = readdir("./name_chunks", join=true)[1] |> readlines

all_results = iterate_similarity_search(politician_names, (:encoded_speech_acts_nm, :encoded_activities_nm), encoded_speech_acts, encoded_activities, 3)

serialize("./search_cache/search_cache_names.jls", all_results)
