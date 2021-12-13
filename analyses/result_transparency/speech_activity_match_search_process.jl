using CSV
using StatsBase
using DataFrames
using DataFramesMeta
using Serialization
using Transducers
using Pipe

@doc """
    sample_rows(speech_act_dataframe::DataFrame, nrow_samples::Int64)::Tuple{Vector{String}, Vector{String}}

Sample `nrow_samples` speech act rows from `speech_act_dataframe`.
"""
function sample_rows(speech_act_dataframe::DataFrame, nrow_samples::Int64)::Tuple{Vector{String}, Vector{String}}
    sampled_rows = sample(1:nrow(speech_act_dataframe), nrow_samples)

    original_queries = speech_act_dataframe[sampled_rows, :original_text]
    most_relevant_search_result = speech_act_dataframe[sampled_rows, :search_results_1]

    original_queries, most_relevant_search_result
end

speech_activity_match_search_results = [deserialize("./search_cache/search_cache_names$(i).jls") for i in 1:6]

filtered_search_results = speech_activity_match_results |> Iterators.flatten |> Filter(!isnothing) |> collect

#[271, 236, 397, 305, 450]
sampled_political_actors = sample(1:length(filtered_search_results), 5)

sampled_search_results = filtered_search_results[sampled_political_actors]

name_sample = @pipe sampled_search_results .|>
                        last .|>
                        getindex(_, :name)

speech_act_dataframes = [first(sampled_search_result) for sampled_search_result in sampled_search_results]

row_samples = [sample_rows(speech_act_dataframe, 3) for speech_act_dataframe in speech_act_dataframes]

serialize("./analyses/result_transparency/name_search_process.jl", row_samples)

name_row_sample = zip(name_sample, row_samples) |> collect
