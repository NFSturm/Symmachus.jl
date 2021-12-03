using CSV
using DataFrames
using Pipe
using DelimitedFiles

include("EncodingUtils.jl")
using .EncodingUtils

@doc """
    validate_name_integrity(encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::Vector{String}

Returns names that are in both `encoded_speech_acts` and `encoded_activities`.
"""
function validate_name_integrity(encoded_speech_acts::DataFrame, encoded_activities::DataFrame)::Vector{String}
    speech_act_names = encoded_speech_acts[:, :actor_name]
    activity_names = encoded_activities[:, :name]

    intersect(speech_act_names, activity_names)
end

encoded_speech_acts = DataFrame(CSV.File("./data/encoded_datasets/speech_acts_encoded.csv"))

transform!(encoded_speech_acts, :encoded_speech_acts_nm => ByRow(x -> parse_encoding(x)) => :encoded_speech_acts_nm)
transform!(encoded_speech_acts, :actor_name => ByRow(x -> lowercase(x)) => :actor_name)

encoded_activities = DataFrame(CSV.File("./data/encoded_datasets/activities_encoded.csv"))

transform!(encoded_activities, :encoded_activities_nm => ByRow(x -> parse_encoding(x)) => :encoded_activities_nm)
transform!(encoded_activities, :name => ByRow(x -> lowercase(x)) => :name)

politician_names = @pipe validate_name_integrity(encoded_speech_acts, encoded_activities) |>
                Set .|>
                String |>
                collect

for i in 0:5
    open("./name_chunks/name_chunk_$(i+1).txt", "w") do f
        writedlm(f, politician_names[(i*103 + 1):((i+1)*103)])
    end
end
