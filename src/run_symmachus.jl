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

	@with_kw mutable struct GeneticArgs # This struct is responsible for the mutation spectrum
	    sentence_context_spectrum::Int64 = 2
	    discourse_context_spectrum::Int64 = 3
	    self_weight_spectrum::Float64 = 0.2
	end
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
    get_labelled_sentences_from_documents(document::Tuple{DataFrame, SymmachusArgs}, labelled_sentences::DataFrame)::DataFrame

Extracts those sentences from embedded `documents` that are in `labelled_sentences` \n
by performing an inner join.
"""
function get_labelled_sentences_from_documents(document::Tuple{DataFrame, SymmachusArgs}, labelled_sentences::DataFrame)::Tuple{DataFrame, SymmachusArgs}
	document_embedded, symmachus_args = document

	select!(document_embedded, Not([:actor_name, :sentence_text]))

	if "discourse_time" ∈ names(labelled_sentences)
		select!(document_embedded, Not([:discourse_time])) # Avoiding duplicate columns downstream
	end

	labelled_sentences = innerjoin(labelled_sentences, document_embedded, on=[:doc_uuid, :sentence_id])#, makeunique=true)
	return labelled_sentences, symmachus_args
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


@everywhere @doc """
    boost(sentence_label_data::Tuple{DataFrame, SymmachusArgs}, boosting_args_array::BoostingArgs)
Trains a booster on `feature_data` and `label_data`. Arguments of the model can be \n
specified using `boosting_args`.
"""
function boost(sentence_label_data::Tuple{DataFrame, SymmachusArgs}, boosting_args::BoostingArgs)
	sentence_label_data, symmachus_args = sentence_label_data
	X_train, y_train, X_test, y_test = make_train_test_data(sentence_label_data, "sentence_embedding", "label", boosting_args.train_prop)

	model_uuid = uuid4() |> string

	@suppress begin
		bst = train_booster(X_train, y_train, arg)
		predictions = predict_booster(bst, X_test)
		confmat = confusion_matrix(predictions, y_test, arg.true_threshold)
		f1_model_score = f1_score(confmat)

		model = Dict(
			:model_id => model_uuid,
			:f1_score => f1_model_score,
			:model_args => arg,
			:symmachus_args => symmachus_args
			)
		)
	end

	sentence_label_data, model
end


@doc """
    broadcast_labels(best_model::Dict{Symbol, Any}, broadcast_targets::DataFrame)::DataFrame

Using the specifications of `best_model`, the labels are broadcast to `broadcast_targets`, \n
which are the datapoints that are newly sampled.
"""
function broadcast_labels(best_model::Dict{Symbol, Any}, label_data::DataFrame, broadcast_targets::DataFrame)::DataFrame

	features = broadcast_targets[!, :sentence_embedding]
	feature_matrix = convert(Array{Float64}, transpose(hcat(features...)))

	train_feature_matrix = convert(Array{Float64}, transpose(hcat(label_data[!, :sentence_embedding]...)))
	train_labels = convert(Array{Int64}, label_data[!, :label])

	@suppress begin
		bst = train_booster(train_feature_matrix, train_labels, best_model[:model_args])
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
	label_data = DataFrame(CSV.File(labelled_data_path))
	transform!(seed_label_data, [:label] .=> ByRow(x -> Float64(x)) .=> [:label_prob])

	# Read unlabelled file names from disk
	files = readdir(unlabelled_data_path)

	@info "Starting embedding-modelling cycle"

	deserialization_items = collect(Set(label_data[!, :doc_uuid])) # Retrieves only unique docs

	deserialization_paths = make_deserialization_paths(deserialization_items)

	documents = retrieve_documents(deserialization_paths)

	sentence_label_dataframes = make_document_dataframe(documents, symmachus_arg)

	sentence_label_data = get_labelled_sentences_from_documents.(sentence_label_dataframes, Ref(label_data))

	@info "Beginning model training. Calling XGBoost... – $(now())"

	# Create the best boosting model for each dataframe
	symmachus_boost_model = boost(sentence_label_data, boosting_args)

	all_docs = retrieve_documents(files) # Actually deserializes the documents

	new_document_dataframe = make_document_dataframe(
		all_docs, symmachus_boost_model[:symmachus_args]) |> concat_dataframes

	all_sentences = broadcast_labels(best_model, embedded_sentences, all_docs)

	all_data_union = concat_dataframes([embedded_sentences, new_sentences])
	select!(new_data_union, Not("sentence_embedding"))

	all_data_union
end

labelled_data_final = train_self("./data/labels/labels.csv", "./data/speech_docs", symmachus_args, boosting_args)
