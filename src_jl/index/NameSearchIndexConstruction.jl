using CSV
using Pipe
using NPZ
using StatsBase
using DataFrames
using DataFramesMeta
using Serialization
using Transducers

include("../EncodingUtils.jl")
using .EncodingUtils

#****************** LOADING DATA ******************

encoded_speech_acts = DataFrame(CSV.File("./data/encoded_datasets/speech_acts_encoded.csv"))

transform!(encoded_speech_acts, :encoded_speech_acts_nm => ByRow(x -> parse_encoding(x)) => :encoded_speech_acts_nm)
transform!(encoded_speech_acts, :actor_name => ByRow(x -> lowercase(x)) => :actor_name)
transform!(encoded_speech_acts, :speech_act_phrases => ByRow(x -> parse_phrases(x)) => :speech_act_phrases)


@doc """
    attach_name(search_result::Tuple{DataFrame, Dict{Symbol, Any}})

Attaches a *name* column to the DataFrame contained in `search_result`.
"""
function attach_name(search_result::Tuple{DataFrame, Dict{Symbol, Any}})
    name = @pipe last(search_result) |> get(_, :name, "")
    insertcols!(first(search_result), :name => name)
    first(search_result)
end

@doc """
    retrieve_embeddings_for_result(search_result::DataFrame, encoded_speech_acts::DataFrame)

Retrieves the original embedding for the original text in `search_result`.
"""
function retrieve_embeddings_for_result(search_result::DataFrame, encoded_speech_acts::DataFrame)
    original_texts = search_result[:, :original_text]
    name = first(search_result[:, :name])

    speech_act_subset = @subset encoded_speech_acts begin
        :actor_name .== name
    end

    embeddings = []

    for speech_act in original_texts
        speech_act_filter = filter(row -> row.sentence_text == speech_act, speech_act_subset)
        push!(embeddings, speech_act_filter.encoded_speech_acts_nm)
    end

    @assert length(embeddings) == length(original_texts)

    search_result[:, :embedding] = embeddings

    search_result
end

@doc """
    retrieve_embeddings_for_all_results(search_results::Vector{DataFrame}, encoded_speech_acts)

A wrapper around *retrieve_embeddings_for_result* to compute for multiple `search_results`.
"""
function retrieve_embeddings_for_all_results(search_results::Vector{DataFrame}, encoded_speech_acts)

    all_dataframes = []

    for search_result in search_results
        push!(all_dataframes, retrieve_embeddings_for_result(search_result, encoded_speech_acts))
    end

    all_dataframes
end

name_search_results_filtered = @pipe [deserialize("./search_cache/search_cache_names$(i).jls") for i in 1:6] |>
                            Iterators.flatten |>
                            Filter(!isnothing) |>
                            attach_name.(_) |>
                            collect

search_results_with_embeddings = retrieve_embeddings_for_all_results(name_search_results_filtered, encoded_speech_acts)

search_results_with_embeddings_concat = vcat(search_results_with_embeddings..., cols=:union)

search_results_with_embeddings_concat[:, :embedding] = [res[1] for res in search_results_with_embeddings_concat[:, :embedding]]

@doc """
    write_numpy_arrays(embeddings::DataFrame, path::String, array_name::String)

Writes out numpy arrays from `embeddings`.
"""
function write_numpy_arrays(embeddings::DataFrame, path::String, array_name::String)
    for i in 1:nrow(embeddings)
        npzwrite("./$path/$(array_name)_$i.npz", embeddings[i, :embedding] |> collect)
    end
end

write_numpy_arrays(@select(search_results_with_embeddings_concat, :embedding), "embedding_arrays", "embedding_index")
