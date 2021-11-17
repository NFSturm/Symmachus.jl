using CSV
using DataFrames
using DataFramesMeta
using Serialization
using Pipe

include("../src_jl/DeputyMetaInfo.jl")
using .MetaInfo

topic_search_results = deserialize("./search_cache/topic_search_cache.jls")

deputy_meta_info = @pipe DataFrame(CSV.File("./analyses/deputies.csv")) |>
        mapcols!(cols -> string.(cols), _)

@doc """
    compute_topic_alignments(topic_search_results, deputy_meta_info::DataFrame, num_topics::Int64, num_results::Int64)

Computes topic alignment scores for every individual, for which search results have been retrieved. \n
Returns the best scoring search results with some meta-information.
"""
function compute_topic_alignments(topic_search_results, deputy_meta_info::DataFrame, num_topics::Int64, num_results::Int64)

    scores = Vector{Tuple{String, String, Float32}}[]

    for topic_num in 1:num_topics
        relevance_info = getindex.(topic_search_results, topic_num) .|> last
        result_container = append_deputy_meta_info(deputy_meta_info, relevance_info)
        best_results = sort(result_container, by = x -> x[:mean_external_alignment], rev=true)[1:num_results]
        result_tuples = [getindex.(Ref(d), (:name, :party, :mean_external_alignment)) for d in best_results]
        push!(scores, result_tuples)
    end
    scores
end

most_aligned_political_actors = compute_topic_alignments(topic_search_results, deputy_meta_info, 17, 10)
