# Self-Training

using XGBoost
using Parameters
using Serialization
using Chain
using Random
using Distributed
using CSV
using DataFrames
using StatsBase

addprocs(5)

@everywhere begin
	using Pkg; Pkg.activate(".")

	using XGBoost
	using Parameters
	using Serialization
	using Chain
	using Random
	using Distributed
	using CSV
	using DataFrames
	using StatsBase
end

@everywhere include("SymmachusCore.jl")
@everywhere include("MLUtils.jl")

@everywhere using .SymmachusCore
@everywhere using .MLUtils

@everywhere begin
	embeddings_vocab, embeddings = load_fasttext_embeddings("./data/embeddings")

	const word_lookup_table = get_word_lookup_table(embeddings_vocab)

	global embeddings_lookup = Lookup(word_lookup_table, embeddings, embeddings_vocab)

	@with_kw mutable struct SymmachusArgs
		max_discourse_context_size::Int64
		max_sentence_context_size::Int64
		self_weight::Number
	end
end

@everywhere symmachus_args = SymmachusArgs(
	max_discourse_context_size=3,
	max_sentence_context_size=3,
	self_weight=0.8
)

@everywhere grid = Dict(:max_discourse_context_size => 1:5, :max_sentence_context_size => 1:5, :self_weight => 0.5:0.1:0.9, :grid_size => 10)

@everywhere symmachus_args = [SymmachusArgs(
	max_discourse_context_size = sample(grid[:max_discourse_context_size], 1) |> first,
	max_sentence_context_size = sample(grid[:max_sentence_context_size], 1) |> first,
	self_weight = sample(grid[:self_weight], 1) |> first
) for i in 1:grid[:grid_size]]


@with_kw mutable struct BoostingArgs
	num_rounds::Int64 # Number of rounds for training the booster
	metrics::Vector{String} # The metric to be chosen
	params::Vector{Pair{String, Any}} # Model parameters
	true_threshold::Float64 # Threshold for positive prediction
	train_prop::Float64 # Proportion of observations to be used for training
end


boosting_args = BoostingArgs(
	num_rounds=150,
	metrics=["aucpr"],
	params= [
		"max_depth" => 2,
		"eta" => 1,
		"objective" => "binary:logistic"
	],
	true_threshold=0.5,
	train_prop=0.8
)

@everywhere @doc """
    make_dataframe_row(sentence::Sentence, embedded_sentence::Vector{Float64})

Creates a DataFrame row by unpacking a Sentence.
"""
function make_dataframe_row(sentence, embedded_sentence::Vector{Float64})
    @unpack doc_uuid, sentence_id, sentence_text, actor_name, discourse_time = sentence
    return doc_uuid, sentence_id, sentence_text, actor_name, discourse_time, embedded_sentence
end


@everywhere @doc """
    doc_to_sent(file::String, symmachus_args:SymmachusArgs)

Using the `embeddings_lookup`, document files are read from the directory and \n
and embedded. The output is a dataframe containing the individual sentences. \n
"""
function doc_to_sent(file::String, symmachus_args::SymmachusArgs)

    embedded_dataframe = DataFrame(
        doc_uuid = String[],
        sentence_id = Int64[],
        sentence_text = String[],
        actor_name = String[],
        discourse_time = String[],
        sentence_embedding = Vector{Float64}[]
    )

    doc = deserialize(file)

    # Unpacking the arguments

    @unpack max_discourse_context_size, max_sentence_context_size, self_weight = symmachus_args

    emb_doc = embed_document(doc, max_discourse_context_size, max_sentence_context_size, self_weight, embeddings_lookup)

    sentences = doc.sentences
    dataframe_rows = make_dataframe_row.(sentences, emb_doc)

    for row in dataframe_rows
        push!(embedded_dataframe, row)
    end

    return embedded_dataframe
end

label_data = DataFrame(CSV.File("./data/labels/labels.csv"))

files = readdir("./data/speech_docs")

deserialization_items = collect(Set(label_data[!, :doc_uuid])) # Retrieves only unique docs

labelled_sentences = label_data[!, [:doc_uuid, :sentence_id]]

make_path_to_speech_docs(doc_name::String) = "./data/speech_docs/" * doc_name * ".jls"


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

deserialization_paths = make_deserialization_paths(deserialization_items)

@doc """
    make_document_dataframe(paths::Vector{String}, symmachus_args::SymmachusArgs)

Deserializes documents and embeds the documents contained in `paths`. \n
These documents are then split into sentences and appended to a dataframe.
"""
function make_document_dataframe(paths::Vector{String}, symmachus_args::SymmachusArgs)
	res = pmap((paths, symmachus_args) -> doc_to_sent(paths, symmachus_args), paths, symmachus_args)
end

@doc """
    concat_dataframes(dataframes::Vector{DataFrame})::DataFrame

A simple function wrapper around `vcat`.
"""
function concat_dataframes(dataframes::Vector{DataFrame})::DataFrame
    vcat(dataframes..., cols=:union)
end

document_dataframe = [make_document_dataframe(deserialization_paths, symmachus_arg) |> concat_dataframes for symmachus_arg in symmachus_args]

sentence_label_data  = innerjoin(labelled_sentences, document_dataframe, on=[:doc_uuid, :sentence_id], makeunique=true)


@doc """
    sample_documents(all_documents_path::String, labelled_documents::Vector{String}, num_documents::Int64)::Vector{String}

Samples documents from a directory. Returns a vector of strings.
"""
function sample_documents(all_documents_path::String, labelled_sentences::Vector{String}, num_documents::Int64)::Vector{String}
	all_documents = readdir(all_documents_path)
	all_documents_id = first.(split.(all_documents, Ref('.')))

	uuid_labels = labelled_sentences[!, :doc_uuid] |> Set |> collect

	labels = labelled_sentences[!, :doc_uuid]

	existing_labels = labelled_sentences[!, :doc_uuid] |> Set |> collect

	document_samples = sample(all_documents_id, 200)

	# Sample only those documents that have not yet been labelled.
	zips = zip(document_samples, document_samples .∈ Ref(existing_labels)) |> collect

	[zip[1] for zip in zips if !zip[2]]
end


@doc """
    train_booster(feature_data::Vector{Vector{Float64}}, label_data::Vector{Int64}}, boosting_args::BoostingArgs)

Trains a boosting classifier.
"""
function train_booster(feature_data::Matrix{Float64}, label_data::Vector{Int64}, boosting_args::BoostingArgs)
	xgboost(feature_data, boosting_args.num_rounds, label=label_data, param = boosting_args.params, metrics = boosting_args.metrics)
end


@doc """
    predict_booster(booster::Booster, test_data)

Predict using a `booster` object.
"""
function predict_booster(booster::Booster, test_data)
	convert(Vector{Float64}, predict(booster, test_data))
end


@doc """
    boost(feature_data::Matrix{Float64}, label_data::Vector{Int64}, boosting_args::BoostingArgs)

Trains a booster on `feature_data` and `label_data`. Arguments of the model can be \n
specified using `boosting_args`.
"""
function boost(feature_data::Matrix{Float64}, label_data::Vector{Int64}, boosting_args::BoostingArgs)
	X_train, y_train, X_test, y_test = make_train_test_data(sentence_label_data, "sentence_embedding", "label", boosting_args.train_prop)
	bst = train_booster(X_train, y_train, boosting_args)
	predictions = predict_booster(bst, X_test)
	confmat = confusion_matrix(predictions, y_test, 0.5)
	f1_model_score = f1_score(confmat)

	return Dict(:f1_score => f1_model_score, :model_args => boosting_args)

end
