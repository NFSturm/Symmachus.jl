using CSV
using DataFrames
using DataFramesMeta
using StatsBase
using Pipe
using Distances
using Serialization
using NPZ

include("EncodingUtils.jl")
using .EncodingUtils

include("kNN.jl")
using .kNN

@doc """
    validate_name_integrity(encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::Vector{String}

Returns names that are in both `encoded_speech_acts` and `encoded_activities`.
"""
function validate_name_integrity(encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::Vector{String}
    speech_act_names = encoded_speech_acts[:, :actor_name]
    activity_names = encoded_activities[:, :name]

    intersect(speech_act_names, activity_names)
end

compute_inner_alignment(speech_act_vector_space::Vector{Float32}, activity_vector_space::Vector{Float32}) = 1 - cosine_dist(speech_act_vector_space, activity_vector_space)

function compute_external_alignment(external_vector_space::Vector{Float32}, speech_act_vector_space::Vector{Float32}, activity_vector_space::Vector{Float32})
    1 .- cosine_dist.(Ref(external_vector_space), [speech_act_vector_space, activity_vector_space])
end

@doc """
    evaluate_topic_search(name::String, encoding_model::Tuple{Symbol, Symbol}, topic_vector::Vector{Float32}, encoded_activities::DataFrame, encoded_speech_acts::DataFrame)

Given a `name` and a `topic_vector`, alignment metrics are computed.
"""
function evaluate_topic_search(name::String, encoding_model::Tuple{Symbol, Symbol}, topic_vector::Vector{Float32}, encoded_activities::DataFrame, encoded_speech_acts::DataFrame)

    # Subsetting speech act data
    speaker_subset = @subset encoded_speech_acts begin
        :actor_name .== name
    end

    # Subsetting deputy activities
    activity_subset = @subset encoded_activities begin
        :name .== name
    end

    activity_data_dense = activity_subset[:, encoding_model[2]] |> collect

    speech_act_data_dense = speaker_subset[:, encoding_model[1]] |> collect

    nearest_activities, _ = compute_nearest_neighbours(topic_vector, activity_data_dense, 10)
    nearest_speech_acts, _  = compute_nearest_neighbours(topic_vector, speech_act_data_dense, 10)

    average_activity_vec_space = mean(activity_subset[nearest_activities, encoding_model[2]])
    average_speech_act_vec_space = mean(speaker_subset[nearest_speech_acts, encoding_model[1]])

    inner_alignment = compute_inner_alignment(average_speech_act_vec_space, average_activity_vec_space)

    external_alignment_speech_acts, external_alignment_activities = compute_external_alignment(
                                                    topic_vector,
                                                    average_speech_act_vec_space,
                                                    average_activity_vec_space)

    alignment_scores = Dict(
        :name => name,
        :inner_alignment => inner_alignment,
        :external_alignment_speech_acts => external_alignment_speech_acts,
        :external_alignment_activities => external_alignment_activities,
        :mean_external_alignment => mean([external_alignment_speech_acts, external_alignment_activities])
    )

    most_aligned_activities = activity_subset[nearest_activities, :text]
    most_aligned_speech_acts = speaker_subset[nearest_speech_acts, :sentence_text]

    most_aligned_activities, most_aligned_speech_acts, alignment_scores
end

@doc """
    evaluate_topics_by_name(name::String, encoding_model::Tuple{Symbol, Symbol}, topic_vectors::Vector{Vector{Float32}}, encoded_activities::DataFrame, encoded_speech_acts::DataFrame)

Evaluates a search model by name on several `topic_vectors`.
"""
function evaluate_topics_by_name(name::String, encoding_model::Tuple{Symbol, Symbol}, topic_vectors::Vector{Vector{Float32}}, encoded_activities::DataFrame, encoded_speech_acts::DataFrame)

    name_topic_metrics = []

    for topic_vector in topic_vectors
        push!(name_topic_metrics, evaluate_topic_search(name, encoding_model, topic_vector, encoded_speech_acts, encoded_activities))
    end

    name_topic_metrics
end

@doc """
    evaluate_all_names(names::Vector{String}, encoding_model::Tuple{Symbol, Symbol}, topic_vectors::Vector{Vector{Float32}}, encoded_activities::DataFrame, encoded_speech_acts::DataFrame)

Wrapper around *evaluate_search_by_topic*. Takes a vector `topic_vectors` to be evaluated.
"""
function evaluate_all_names(names::Vector{String}, encoding_model::Tuple{Symbol, Symbol}, topic_vectors::Vector{Vector{Float32}}, encoded_activities::DataFrame, encoded_speech_acts::DataFrame)

    name_results = []

    for name in names
        push!(name_results, evaluate_topics_by_name(name, encoding_model, topic_vectors, encoded_speech_acts, encoded_activities))
    end

    name_results

end

unpack_numpy_array(path::String) = @pipe values(npzread(path)) |> collect |> getindex(_, 1)

topic_vector_paths = readdir(joinpath("./src_py/pt_arrays"), join=true)

orderings = @pipe split.(topic_vector_paths, "sdg") .|> split(last(_), ".") .|> first .|> parse(Int, _)

ordered_paths = @pipe zip(topic_vector_paths, orderings) |> collect |> sort(_, by= x -> x[2]) |> getindex.(_, 1)

topic_vectors = unpack_numpy_array.(ordered_paths)

encoded_activities = DataFrame(CSV.File("./data/encoded_datasets/activities_encoded.csv"))
transform!(encoded_activities, :encoded_activities_pt => ByRow(x -> parse_encoding(x)) => :encoded_activities_pt)
transform!(encoded_activities, :name => ByRow(x -> lowercase(x)) => :name)

encoded_speech_acts = DataFrame(CSV.File("./data/encoded_datasets/speech_acts_encoded.csv"))
transform!(encoded_speech_acts, :encoded_speech_acts_pt => ByRow(x -> parse_encoding(x)) => :encoded_speech_acts_pt)
transform!(encoded_speech_acts, :actor_name => ByRow(x -> lowercase(x)) => :actor_name)

politician_names = @pipe validate_name_integrity(encoded_speech_acts, encoded_activities) |>
                Set .|>
                String |>
                collect

topic_results = evaluate_all_names(politician_names, (:encoded_speech_acts_pt, :encoded_activities_pt), topic_vectors, encoded_activities, encoded_speech_acts)

serialize("./search_cache/topic_search_cache.jls", topic_results)
