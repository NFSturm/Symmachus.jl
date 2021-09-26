# Self-Training

using XGBoost
using XGBoost: predict as apply_booster

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
using Revise

addprocs(5)

@everywhere begin
	using Pkg; Pkg.activate(".")

	using XGBoost
	using XGBoost: predict as apply_booster

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
	using Revise
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

@everywhere grid = Dict(:max_discourse_context_size => 1:5, :max_sentence_context_size => 1:5, :self_weight => 0.5:0.1:0.9, :grid_size => 10)

@everywhere symmachus_args_array = [SymmachusArgs(
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

labelled_sentences = label_data[!, [:doc_uuid, :sentence_id, :label]]

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

documents = retrieve_documents(deserialization_paths)

@doc """
    make_document_dataframe(paths::Vector{Document}, symmachus_args::SymmachusArgs)
Deserializes documents and embeds the documents contained in `paths`. \n
These documents are then split into sentences and appended to a dataframe. \n
Arguments for the *Symmachus* embedding can be specified using `args`.
"""
function make_document_dataframe(docs::Vector{Document}, args::SymmachusArgs)
	res = pmap((d, arg) -> doc_to_sent(d, arg), docs, Iterators.repeated(args, length(docs)) |> collect)
end

@doc """
    concat_dataframes(dataframes::Vector{DataFrame})::DataFrame
A simple function wrapper around `vcat`.
"""
function concat_dataframes(dataframes::Vector{DataFrame})::DataFrame
    vcat(dataframes..., cols=:union)
end


document_dataframe = [(make_document_dataframe(documents, symmachus_arg) |> concat_dataframes, symmachus_arg) for symmachus_arg in symmachus_args_array]

serialize("./cache/document_dataframe.jls", document_dataframe)

@doc """
    get_labelled_sentences_from_documents(document::Tuple{DataFrame, SymmachusArgs}, labelled_sentences::DataFrame)::DataFrame

Extracts those sentences from embedded `documents` that are in `labelled_sentences` \n
by performing an inner join.
"""
function get_labelled_sentences_from_documents(document::Tuple{DataFrame, SymmachusArgs}, labelled_sentences::DataFrame)::Tuple{DataFrame, SymmachusArgs}
	document_embedded, symmachus_args = document
	labelled_sentences = innerjoin(labelled_sentences, document_embedded, on=[:doc_uuid, :sentence_id], makeunique=true)
	return labelled_sentences, symmachus_args
end

sentence_label_data = get_labelled_sentences_from_documents.(document_dataframe, Ref(labelled_sentences))


@doc """
    train_booster(feature_data::Vector{Vector{Float64}}, label_data::Vector{Int64}}, boosting_args::BoostingArgs)
Trains a boosting classifier.
"""
function train_booster(feature_data::Matrix{Float64}, label_data::Vector{Int64}, boosting_args::BoostingArgs)
	booster = xgboost(feature_data, boosting_args.num_rounds, label=label_data, param = boosting_args.params, metrics = boosting_args.metrics)
	return booster
end

@doc """
    predict_booster(booster::Booster, test_data)
Predict using a `Booster` object.
"""
function predict_booster(booster::Booster, test_data)
	convert(Vector{Float64}, apply_booster(booster, test_data))
end


@doc """
    boost(sentence_label_data::Tuple{DataFrame, SymmachusArgs}, boosting_args::BoostingArgs)
Trains a booster on `feature_data` and `label_data`. Arguments of the model can be \n
specified using `boosting_args`.
"""
function boost(sentence_label_data::Tuple{DataFrame, SymmachusArgs}, boosting_args::BoostingArgs)
	sentence_label_data, symmachus_args = sentence_label_data
	X_train, y_train, X_test, y_test = make_train_test_data(sentence_label_data, "sentence_embedding", "label", boosting_args.train_prop)
	bst = train_booster(X_train, y_train, boosting_args)
	predictions = predict_booster(bst, X_test)
	confmat = confusion_matrix(predictions, y_test, boosting_args.true_threshold)
	f1_model_score = f1_score(confmat)

	return Dict(
		:model_id => uuid4() |> string,
		:f1_score => f1_model_score,
		:model => bst,
		:model_args => boosting_args,
		:symmachus_args => symmachus_args,
		:predictions => predictions
	)
end


@doc """
    sample_documents(all_documents_path::String, labelled_documents::Vector{String}, num_documents::Int64)::Vector{String}
Samples documents from a directory. Returns a vector of strings.
"""
function sample_documents(all_documents_path::String, labelled_sentences::DataFrame, num_documents::Int64)::Vector{String}
	all_documents = readdir(all_documents_path)
	all_documents_id = first.(split.(all_documents, Ref('.')))

	uuid_labels = labelled_sentences[!, :doc_uuid] |> Set |> collect

	labels = labelled_sentences[!, :doc_uuid]

	existing_labels = labelled_sentences[!, :doc_uuid] |> Set |> collect

	document_samples = sample(all_documents_id, num_documents)

	# Sample only those documents that have not yet been labelled.
	zips = zip(document_samples, document_samples .∈ Ref(existing_labels)) |> collect

	[first(zip) for zip in zips if !last(zip)]
end

new_deserialization_paths = sample_documents("./data/speech_docs", labelled_sentences, 50) |> make_deserialization_paths

new_documents = retrieve_documents(new_deserialization_paths)

new_document_dataframes = [make_document_dataframe(new_deserialization_paths, symmachus_arg) |> concat_dataframes for symmachus_arg in symmachus_args_array]

function broadcast_labels(model::Booster, broadcast_targets::DataFrame)::DataFrame
	features = broadcast_targets[!, :sentence_embedding]
	feature_labels = predict_booster(model, features)
	broadcast_targets[!, :labels] = feature_labels

	feature_labels_sorted_rev = sort(broadcast_targets, by=:labels, rev=true)
	feature_labels_sorted = sort(broadcast_targets, by=:labels)

	confident_predictions_rev = feature_labels_sorted_rev[1:20]
	confident_predictions = feature_labels_sorted[1:20]

	confident_predictions_all = concat_dataframes([confident_predictions_rev, confident_predictions])

	transform(confident_predictions_all, [:labels] .=> ByRow(x -> round(x)) .=> [:labels])

end
