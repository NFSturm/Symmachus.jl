using Pkg

Pkg.activate(".")

using CSV
using DataFrames
using Pipe
using DataFramesMeta
using StatsBase
using Serialization
using Distances
using Revise
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

    inner_alignment, external_alignment_speech_acts, external_alignment_activities, mean([external_alignment_speech_acts, external_alignment_activities])
end

retrieve_topic_vector_paths(model_names::Vector{String}, path::String) = [readdir(joinpath(path, model_name * "_arrays"), join=true) for model_name in model_names]

sample_names(names::Vector{String}, number_of_names::Int64) = sample(names, number_of_names)

@doc """
    evaluate_search_by_topic(names::Vector{String}, encoding_model::Tuple{Symbol, Symbol}, topic_vector::Vector{Float32}, encoded_activities::DataFrame, encoded_speech_acts::DataFrame)

Evaluates a search model by topic on several `names`.
"""
function evaluate_search_by_topic(names::Vector{String}, encoding_model::Tuple{Symbol, Symbol}, topic_vector::Vector{Float32}, encoded_activities::DataFrame, encoded_speech_acts::DataFrame)

    metrics = []

    for name in names
        push!(metrics, evaluate_topic_search(name, encoding_model, topic_vector, encoded_speech_acts, encoded_activities))
    end

    average_inner_alignment = mean(getindex.(metrics, 1))
    external_alignment_speech_acts = mean(getindex.(metrics, 2))
    external_alignment_activities = mean(getindex.(metrics, 3))
    aggregate_external_alignment = mean([external_alignment_activities, external_alignment_speech_acts])

    average_inner_alignment, external_alignment_speech_acts, external_alignment_activities, aggregate_external_alignment
end

@doc """
    evaluate_all_topics(names::Vector{String}, encoding_model::Tuple{Symbol, Symbol}, topic_vectors::Vector{Vector{Float32}}, encoded_activities::DataFrame, encoded_speech_acts::DataFrame)

Wrapper around *evaluate_search_by_topic*. Takes a vector `topic_vectors` to be evaluated.
"""
function evaluate_all_topics(names::Vector{String}, encoding_model::Tuple{Symbol, Symbol}, topic_vectors::Vector{Vector{Float32}}, encoded_activities::DataFrame, encoded_speech_acts::DataFrame)

    topic_stats = []

    for topic_vector in topic_vectors
        push!(topic_stats, evaluate_search_by_topic(names, encoding_model, topic_vector, encoded_speech_acts, encoded_activities))
    end

    average_inner_alignment = mean(getindex.(topic_stats, 1))
    external_alignment_speech_acts = mean(getindex.(topic_stats, 2))
    external_alignment_activities = mean(getindex.(topic_stats, 3))
    aggregate_external_alignment = mean([external_alignment_activities, external_alignment_speech_acts])

    average_inner_alignment, external_alignment_speech_acts, external_alignment_activities, aggregate_external_alignment
end

@info "Loading activities…"

encoded_activities = DataFrame(CSV.File("./data/encoded_datasets/activities_encoded.csv"))

transform!(encoded_activities, :encoded_activities_ml => ByRow(x -> parse_encoding(x)) => :encoded_activities_ml)
transform!(encoded_activities, :encoded_activities_pt => ByRow(x -> parse_encoding(x)) => :encoded_activities_pt)
transform!(encoded_activities, :encoded_activities_nm => ByRow(x -> parse_encoding(x)) => :encoded_activities_nm)

transform!(encoded_activities, :name => ByRow(x -> lowercase(x)) => :name)

transform!(encoded_activities, :activity_phrases => ByRow(x -> parse_phrases(x)) => :activity_phrases)

@info "Loading speech acts…"

encoded_speech_acts = DataFrame(CSV.File("./data/encoded_datasets/speech_acts_encoded.csv"))

transform!(encoded_speech_acts, :encoded_speech_acts_ml => ByRow(x -> parse_encoding(x)) => :encoded_speech_acts_ml)
transform!(encoded_speech_acts, :encoded_speech_acts_pt => ByRow(x -> parse_encoding(x)) => :encoded_speech_acts_pt)
transform!(encoded_speech_acts, :encoded_speech_acts_nm => ByRow(x -> parse_encoding(x)) => :encoded_speech_acts_nm)

transform!(encoded_speech_acts, :actor_name => ByRow(x -> lowercase(x)) => :actor_name)

transform!(encoded_speech_acts, :speech_act_phrases => ByRow(x -> parse_phrases(x)) => :speech_act_phrases)

@info "Retrieving topic vectors…"

topic_vector_paths = retrieve_topic_vector_paths(["ml", "pt", "nm"], "./src_py")

unpack_numpy_array(path::String) = @pipe values(npzread(path)) |> collect |> getindex(_, 1)

topic_vectors_unpacked = [unpack_numpy_array.(topic_vector_path) for topic_vector_path in topic_vector_paths]

@info "Sampling names…"

politician_names = @pipe validate_name_integrity(encoded_speech_acts, encoded_activities) |>
                Set .|>
                String |>
                collect |>
                sample_names(_, 100)

@info "Evaluating topic vector model…"

topic_result_ml = evaluate_all_topics(politician_names, (:encoded_speech_acts_ml, :encoded_activities_ml), topic_vectors_unpacked[1], encoded_activities, encoded_speech_acts)

topic_result_pt = evaluate_all_topics(politician_names, (:encoded_speech_acts_pt, :encoded_activities_pt), topic_vectors_unpacked[2], encoded_activities, encoded_speech_acts)

topic_result_nm = evaluate_all_topics(politician_names, (:encoded_speech_acts_nm, :encoded_activities_nm), topic_vectors_unpacked[3], encoded_activities, encoded_speech_acts)

serialize("./search_cache/topic_validation_results.jls",
    Dict(
        :topic_result_ml => topic_result_ml,
        :topic_result_pt => topic_result_pt,
        :topic_result_nm => topic_result_nm
        )
    )
