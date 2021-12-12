using CSV
using StatsBase
using DataFrames
using DataFramesMeta
using Serialization

topic_search_results = deserialize("./search_cache/topic_search_cache.jls")

#[348, 607, 355, 155, 321]
politician_samples = sample(1:length(topic_search_results), 5)

serialize(joinpath(@__DIR__, "politician_samples.jls"), politician_samples)

topic_search_results_samples = topic_search_results[politician_samples]

serialize(joinpath(@__DIR__, "topic_search_results_samples.jls"), topic_search_results_samples)
