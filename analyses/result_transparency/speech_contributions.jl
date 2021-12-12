using CSV
using DataFrames
using DataFramesMeta
using StatsBase
using Pipe
using Distances
using Serialization
using NPZ

include("../../src_jl/EncodingUtils.jl")
using .EncodingUtils

original_texts = deserialize("./analyses/result_transparency/topic_search_results_samples.jls")

encoded_speech_acts = DataFrame(CSV.File("./data/encoded_datasets/speech_acts_encoded.csv"))
transform!(encoded_speech_acts, :actor_name => ByRow(x -> lowercase(x)) => :actor_name)
transform!(encoded_speech_acts, :speech_act_phrases => ByRow(x -> parse_phrases(x)) => :speech_act_phrases)

@doc """
    compute_text_statistics(topic_search_result, encoded_speech_acts::DataFrame)

Computes corpus statistics for `topic_search_result`, using the most aligned SDG.
"""
function compute_text_statistics(topic_search_result, encoded_speech_acts::DataFrame)
    order_nums = [(data, i) for (i, data) in enumerate(topic_search_result)]

    most_aligned_texts = @pipe order_nums |>
                                sort(_, by = x -> x[1][end][:mean_external_alignment], rev=true) |>
                                first |>
                                first |>
                                _[2]

    most_aligned_sdg = @pipe order_nums |>
                                sort(_, by = x -> x[1][end][:mean_external_alignment], rev=true) |>
                                first |>
                                last

    sdg_score_orderings = @pipe order_nums |>
                                sort(_, by = x -> x[1][end][:mean_external_alignment], rev=true) |>
                                last.(_)

    deputy_name = @pipe topic_search_result |>
                        first |>
                        last |>
                        _[:name]

    speech_act_subset = @subset encoded_speech_acts begin
        :actor_name .== deputy_name
    end

    rows = []

    for speech_act in most_aligned_texts
        speech_act_filter = filter(row -> row.sentence_text == speech_act, speech_act_subset)
        push!(rows, speech_act_filter)
    end

    speech_act_subset_without_aligned_texs = @rsubset encoded_speech_acts begin
        :actor_name == deputy_name
        :sentence_text ∉ most_aligned_texts
    end

    phrases_most_aligned_texts = @pipe vcat(rows...) |>
                                    _[:, :speech_act_phrases] |>
                                    Iterators.flatten |>
                                    countmap |>
                                    sort(_, byvalue=true, rev=true)

    phrases_corpus = @pipe speech_act_subset_without_aligned_texs[:, :speech_act_phrases] |>
                        Iterators.flatten |>
                        countmap |>
                        sort(_, byvalue=true, rev=true)

    phrases_corpus, phrases_most_aligned_texts, most_aligned_texts, deputy_name, most_aligned_sdg, sdg_score_orderings
end

corpus_statistics = [compute_text_statistics(search_result, encoded_speech_acts) for search_result in original_texts]
