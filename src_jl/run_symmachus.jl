# Running the model for the entire corpus
using Pkg; Pkg.activate(".")

using XGBoost
using XGBoost: predict as apply_booster
using MLStyle.Modules.Cond

using ThreadsX

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

addprocs(5)

@everywhere begin
	using Pkg; Pkg.activate(".")

	using XGBoost
	using XGBoost: predict as apply_booster

	using MLStyle.Modules.Cond

	using ThreadsX

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
end

@everywhere include("SymmachusCore.jl")
@everywhere include("MLUtils.jl")

@everywhere using .SymmachusCore
@everywhere using .MLUtils


@everywhere begin
	embeddings_vocab, embeddings = load_fasttext_embeddings("./data/embeddings")

	const word_lookup_table = get_word_lookup_table(embeddings_vocab)

	global embeddings_lookup = Lookup(word_lookup_table, embeddings, embeddings_vocab) # FIXME: This is a memory bottleneck

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

	symmachus_args = @pipe deserialize("./cache/final_model/model_history.jls") |> last |> _[:symmachus_args]
	boosting_args = @pipe deserialize("./cache/final_model/model_history.jls") |> last |> _[:boosting_args]

end


@everywhere @doc """
    make_dataframe_rows(sentences::Vector{Sentence}, embedded_sentences::Vector{Vector{Float64}})

Creates DataFrame rows by unpacking `sentences` and appending `embedded_sentences` one by one.
"""
function make_dataframe_rows(sentences::Vector{Sentence}, embedded_sentences::Vector{Vector{Float64}})

	rows = []

	for index in eachindex(sentences)
		@unpack doc_uuid, sentence_id, sentence_text, actor_name, discourse_time = sentences[index]
		push!(rows, [doc_uuid, sentence_id, sentence_text, actor_name, discourse_time, embedded_sentences[index]])
	end
    rows
end


@doc """
    retrieve_documents(paths::Vector{String})

Retrieves documents of type *Document* from `paths`.
"""
function retrieve_documents(paths::Vector{String})

	documents = Document[]

	docs = ThreadsX.foreach(paths) do p
		doc = deserialize(p)
		push!(documents, doc)
	end
	return documents
end

@everywhere @doc """
    doc_to_sent(doc::Document, symmachus_args::SymmachusArgs)
Using the `embeddings_lookup`, document files are read from the directory and \n
and embedded. The output is a dataframe containing the individual sentences. \n
"""
function doc_to_sent(doc::Document, symmachus_args::SymmachusArgs)

    embedded_dataframe = DataFrame(
        doc_uuid = String[],
        sentence_id = Int64[],
        sentence_text = String[],
        actor_name = String[],
        discourse_time = String[],
        sentence_embedding = Vector{Float64}[]
    )

    # Unpacking the arguments

    @unpack max_discourse_context_size, max_sentence_context_size, self_weight = symmachus_args

    emb_doc = embed_document(doc, max_discourse_context_size, max_sentence_context_size, self_weight, embeddings_lookup)

	dataframe_rows = make_dataframe_rows(doc.sentences, emb_doc)

    for row in dataframe_rows
        push!(embedded_dataframe, row)
    end

	@info "Embedded document – $(now())"
    return embedded_dataframe
end


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

make_path_to_speech_docs(doc_name::String) = "./data/speech_docs/" * doc_name * ".jls"

@doc """
    make_document_dataframe(docs::Vector{Document}, args::SymmachusArgs)

Deserializes documents and embeds the documents contained in `paths`. \n
These documents are then split into sentences and appended to a dataframe. \n
Arguments for the *Symmachus* embedding can be specified using `args`.
"""
function make_document_dataframe(docs::Vector{Document}, args::SymmachusArgs)
	res = pmap((d, arg) -> doc_to_sent(d, arg), docs, Iterators.repeated(args, length(docs)) |> collect)
	@info "Embedded document DataFrame – $(now())"
	res
end

@doc """
    concat_dataframes(dataframes::Vector{DataFrame})::DataFrame
A simple function wrapper around `vcat`.
"""
function concat_dataframes(dataframes::Vector{DataFrame})::DataFrame
    vcat(dataframes..., cols=:union)
end

#@everywhere sentence_label_data = deserialize("./cache/sentence_label_data.jls")

@doc """
    get_labelled_sentences_from_documents(document::DataFrame, labelled_sentences::DataFrame)::DataFrame

Extracts those sentences from embedded `documents` that are in `labelled_sentences` \n
by performing an inner join.
"""
function get_labelled_sentences_from_documents(document::DataFrame, labelled_sentences::DataFrame)::DataFrame

	select!(document, Not([:actor_name, :sentence_text]))

	if "discourse_time" ∈ names(labelled_sentences)
		select!(document, Not([:discourse_time])) # Avoiding duplicate columns downstream
	end

	labelled_sentences = innerjoin(labelled_sentences, document, on=[:doc_uuid, :sentence_id])#, makeunique=true)
	return labelled_sentences
end

@everywhere @doc """
    train_booster(feature_data::Vector{Vector{Float64}}, label_data::Vector{Int64}}, boosting_args::BoostingArgs)
Trains a boosting classifier.
"""
function train_booster(feature_data::Matrix{Float64}, label_data::Vector{Int64}, boosting_args::BoostingArgs)
	booster = xgboost(feature_data, boosting_args.num_rounds, label=label_data, param = boosting_args.params, metrics = boosting_args.metrics)
	return booster
end

@everywhere @doc """
    predict_booster(booster::Booster, test_data)
Predict using a `Booster` object.
"""
function predict_booster(booster::Booster, test_data)
	convert(Vector{Float64}, apply_booster(booster, test_data))
end


@doc """
    broadcast_labels(best_model::Dict{Symbol, Any}, broadcast_targets::DataFrame)::DataFrame

Using the specifications of `best_model`, the labels are broadcast to `broadcast_targets`, \n
which are the datapoints that are newly sampled.
"""
function broadcast_labels(args::BoostingArgs, label_data::DataFrame, broadcast_targets::DataFrame)::DataFrame

	features = broadcast_targets[!, :sentence_embedding]
	feature_matrix = convert(Array{Float64}, transpose(hcat(features...)))

	train_feature_matrix = convert(Array{Float64}, transpose(hcat(label_data[!, :sentence_embedding]...)))
	train_labels = convert(Array{Int64}, label_data[!, :label])

	@suppress begin
		bst = train_booster(train_feature_matrix, train_labels, args)
		feature_labels = predict_booster(bst, feature_matrix)
		broadcast_targets[!, :label] = feature_labels
	end

	transform!(broadcast_targets, [:label] .=> ByRow(x -> Float64(x)) .=> [:label_prob]) # We annotate the label to check the certainty of the model
	transform!(broadcast_targets, [:label] .=> ByRow(x -> round(x) |> Int) .=> [:label])

	return broadcast_targets
end


@doc """
    run_symmachus(labelled_data_path::String, unlabelled_data_path::String, symmachus_args::SymmachusArgs, boosting_args::BoostingArgs)

Trains the XGBoost model on existing, labelled data in `labelled_data_path`. This model then infers labels for all documents in `unlabelled_data_path`.
"""
function run_symmachus(labelled_data_path::String, unlabelled_data_path::String, symmachus_args::SymmachusArgs, boosting_args::BoostingArgs)

	# Read labelled data from disk
	label_data = deserialize(labelled_data_path)

	# Read unlabelled file names from disk
	files = readdir(unlabelled_data_path)

	@info "Starting embedding-modelling cycle"

	deserialization_items = collect(Set(label_data[!, :doc_uuid])) # Retrieves only unique docs

	deserialization_paths = make_deserialization_paths(deserialization_items)

	documents = retrieve_documents(deserialization_paths)

	sentence_label_dataframe = make_document_dataframe(documents, symmachus_args) |> concat_dataframes

	@info "Embedded training label data – $(now())"

	sentence_label_data = get_labelled_sentences_from_documents(sentence_label_dataframe, label_data)

	all_docs = retrieve_documents(readdir(unlabelled_data_path, join=true)) # Deserializes the entire corpus

	@info "Embedding unlabelled data – $(now())"

	new_document_dataframe = make_document_dataframe(
		all_docs, symmachus_args) |> concat_dataframes

	@info "Document embedding completed – $(now())"

	@info "Broadcasting labels – $(now())"

	all_sentences = broadcast_labels(boosting_args, sentence_label_data, new_document_dataframe)

	select!(all_sentences, Not("sentence_embedding"))

	all_sentences
end

labelled_data_final = run_symmachus("./cache/final_model/labelled_data_final.jls", "./data/speech_docs", symmachus_args, boosting_args)
