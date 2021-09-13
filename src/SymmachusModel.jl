# Self-Training

using XGBoost
using DataFrames
using Parameters
using Serialization
using Chain
using Random
using Distributed

addprocs(5)

@everywhere begin
	using Pkg; Pkg.activate(".")

	using XGBoost
	using DataFrames
	using Parameters
	using Serialization
	using Chain
	using Random
	using Distributed
end

@everywhere include("SymmachusCore.jl")
@everywhere using .SymmachusCore

@everywhere begin
	embeddings_vocab, embeddings = load_fasttext_embeddings("./data/embeddings")

	const word_lookup_table = get_word_lookup_table(embeddings_vocab)

	global embeddings_lookup = Lookup(word_lookup_table, embeddings, embeddings_vocab)

	@with_kw mutable struct ModelArgs
		max_discourse_context_size::Int64
		max_sentence_context_size::Int64
		self_weight::Number
	end
end

@everywhere global symmachus_args = ModelArgs(
	max_discourse_context_size=3,
	max_sentence_context_size=3,
	self_weight=0.8
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
    doc_to_sent(file::String)

Using the `embeddings_lookup`, document files are read from the directory and \n
and embedded. The output is a dataframe containing the individual sentences. \n
"""
function doc_to_sent(file::String)

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

    @unpack max_discourse_context, max_sentence_context, self_weight = args

    emb_doc = embed_document(doc, max_discourse_context, max_sentence_context, self_weight, embeddings_lookup)

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
    make_document_dataframe(paths::Vector{String}, model_args)

Deserializes documents and embeds the documents contained in `paths`. \n
These documents are then split into sentences and appended to a dataframe. \n
Parameters of the *SymmachusModel* can be specified by `model_args`.
"""
function make_document_dataframe(paths::Vector{String})
    res = pmap(doc_to_sent, paths)
end


@doc """
    concat_dataframes(dataframes::Vector{DataFrame})::DataFrame

A simple function wrapper around `vcat`.
"""
function concat_dataframes(dataframes::Vector{DataFrame})::DataFrame
    vcat(dataframes..., cols=:union)
end

document_dataframe = make_document_dataframe(deserialization_paths, model_args) |> concat_dataframes

sentence_label_data  = innerjoin(label_data, document_dataframe, on=[:doc_uuid, :sentence_id], makeunique=true)

@doc """
    make_train_test_data(all_data::DataFrame, label_column::String, feature_column::String, train_prop::Float64)

Randomly shuffles `all_data` according to `train_prop`. Returns the training and test data including the corresponding labels.
"""
function make_train_test_data(all_data::DataFrame, feature_column::String, label_column::String, train_prop::Float64)

	all_labels = all_data[!, label_column]
	all_features = all_data[!, feature_column]

	data_length = nrow(all_data)

	indices = shuffle(1:1:data_length)

	split_index = Int(floor(data_length * train_prop))

	train_indices = indices[1:split_index]
	test_indices = indices[split_index:end]

	X_train, y_train = all_features[train_indices], all_labels[train_indices]
	X_test, y_test = all_features[test_indices], all_labels[test_indices]

	return X_train, y_train, X_test, y_test

end

X_train, y_train, X_test, y_test = make_train_test_data(sentence_label_data, "sentence_embedding", "label", 0.8)

@with_kw mutable struct BoostingArgs
	num_rounds::Int64
	nfold::Int64
	metrics::String
	params::Vector{Pair{String, Any}}
end


boosting_args = BoostingArgs(
	num_rounds=50,
	nfold=2,
	metrics="precision",
	params= [
		"max_depth" => 2,
		"eta" => 1,
		"objective" => "binary:logistic"
	]
)


@doc """
    train_booster(feature_data::Vector{Vector{Float64}}, label_data::Vector{Int64}}, boosting_args)

Trains a boosting classifier.
"""
function train_booster(feature_data::Vector{Vector{Float64}}, label_data::Vector{Int64}}, boosting_args)

	nfold_cv(feature_data, boosting_args.num_rounds, boosting_args.nfold, label = label_data, param = param, metrics = boosting_args.metrics)

end


@doc """
    sample_documents(all_documents_path::String, labelled_documents::Vector{String})::Vector{String}

Samples documents from a directory. Returns a vector of strings.
"""
function sample_documents(all_documents_path::String, labelled_sentences::Vector{String})::Vector{String}
	all_documents = readdir(all_documents_path)
	all_documents_id = first.(split.(all_documents, Ref('.')))

	uuid_labels = labelled_sentences[!, :doc_uuid] |> Set |> collect

	# Do an anti-join with existing labels after deserialization

end



all_documents = readdir("./data/speech_docs")

all_documents_id = first.(split.(documents, Ref('.')))

labels = labelled_sentences[!, :doc_uuid]

labelled_sentences[!, :doc_uuid] |> Set |> collect

docu
