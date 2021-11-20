using Pkg

Pkg.activate(".")

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

include("TopicSearchFunctions.jl")
using .TopicSearch

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
