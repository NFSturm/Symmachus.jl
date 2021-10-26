using Parameters
using Serialization
using Chain
using Random
using Distributed
using CSV
using DataFrames
using StatsBase
using UUIDs
using Suppressor
using Pipe
using Revise
using Dates

@with_kw mutable struct SymmachusArgs
    max_discourse_context_size::Int64
    max_sentence_context_size::Int64
    self_weight::Number
end

@with_kw mutable struct BoostingArgs
    num_rounds::Int64 # Number of rounds for training the booster
    metrics::Vector{String} # The metric to be chosen
    params::Vector{Pair{String, Any}} # Model parameters
    true_threshold::Float64 # Threshold for positive prediction
    train_prop::Float64 # Proportion of observations to be used for training
end

label_data = deserialize("./cache/final_model/labelled_data_final.jls")

model_history = deserialize("./cache/cache/final_model/model_history.jls")

deserialization_items = collect(Set(label_data[!, :doc_uuid]))

@doc """
    make_deserialization_paths(items::Vector{String})
Create deserialization paths for documents.
"""
function make_deserialization_paths(items::Vector{String})

	paths = String[]

	foreach(items) do item
		path_name = make_path_to_speech_docs(item)
		push!(paths, path_name)
	end

	return paths

end

@doc """
    retrieve_documents(paths::Vector{String})

Retrieves documents of type *Document* from `paths`.
"""
function retrieve_documents(paths::Vector{String})

	documents = Document[]

	docs = foreach(paths) do p
		doc = deserialize(p)
		push!(documents, doc)
	end
	return documents
end

make_path_to_speech_docs(doc_name::String) = "./data/speech_docs/" * doc_name * ".jls"

include("SymmachusCore.jl")
using .SymmachusCore

deserialization_items = collect(Set(label_data[!, :doc_uuid])) # Retrieves only unique docs

deserialization_paths = make_deserialization_paths(deserialization_items)

using BenchmarkTools

@btime retrieve_documents(deserialization_paths)

if nrow(label_data) > 150
    println("yeeees")
end

CSV.write("./cache/label_data_test.csv", label_data)

best_args = deserialize("/home/nfsturm/Dev/Symmachus.jl/cache/model_specs_20211010_181124.jls")

using BSON

bson("./cache/temp_cache/model_specs_test.bson", best_args)

model_history = deserialize("/home/nfsturm/Dev/Symmachus.jl/cache/final_model/model_history.jls")

bson("./cache/temp_cache/model_history_test.bson", model_history, @__MODULE__)

best_args

df = DataFrame(CSV.File("./cache/temp_cache/label_data_test.csv"))


select!(df, Not(r"_\d+|sentence_embedding"))

names(df)
