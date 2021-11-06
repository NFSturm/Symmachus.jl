using CSV
using DataFrames
using DataFramesMeta
using StatsBase
using Distances
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
