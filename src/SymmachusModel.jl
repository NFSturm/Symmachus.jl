# Self-Training

include("LabelingUtils.jl")
include("SymmachusCore.jl")

using XGBoost
using DataFrames
using FLoops
using Parameters
using ThreadsX
using Serialization
using Chain
using Random

using .LabelingUtils
using .SymmachusCore

embeddings_vocab, embeddings = load_fasttext_embeddings("./data/embeddings")

const word_lookup_table = get_word_lookup_table(embeddings_vocab)

global embeddings_lookup = Lookup(word_lookup_table, embeddings, embeddings_vocab)

@with_kw mutable struct ModelArgs
	max_discourse_context_size::Int64
	max_sentence_context_size::Int64
	self_weight::Number
end

model_args = ModelArgs(
	max_discourse_context_size=3,
	max_sentence_context_size=3,
	self_weight=0.8
)

@doc """
    make_dataframe_row(sentence::Sentence, embedded_sentence::Vector{Float64})

Creates a DataFrame row by unpacking a Sentence.
"""
function make_dataframe_row(sentence, embedded_sentence::Vector{Float64})
    @unpack doc_uuid, sentence_id, sentence_text, actor_name, discourse_time = sentence
    return doc_uuid, sentence_id, sentence_text, actor_name, discourse_time, embedded_sentence
end


@doc """
    doc_to_sent(embeddings_lookup::Lookup, args)

Using the `embeddings_lookup`, document files are read from the directory and \n
and embedded. The output is a dataframe containing the individual sentences. \n
`Args` defines a parameter struct.
"""
function doc_to_sent(file::String, embeddings_lookup::Lookup, args)

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

    @unpack max_discourse_context_size, max_sentence_context_size, self_weight = args

    emb_doc = embed_document(doc, max_discourse_context_size, max_sentence_context_size, self_weight, embeddings_lookup)

    sentences = doc.sentences
    dataframe_rows = make_dataframe_row.(sentences, emb_doc)

    for row in dataframe_rows
        push!(embedded_dataframe, row)
    end

    return embedded_dataframe
end

label_data = read_label_data("./data/labels/labels.csv")

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

	ThreadsX.foreach(items) do item
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
Parameters of the **SymmachusModel** can be specified by `model_args`.
"""
function make_document_dataframe(paths::Vector{String}, model_args)

	sentences_data = DataFrame[]

	@unpack max_discourse_context_size, max_sentence_context_size, self_weight = model_args

	ThreadsX.foreach(paths) do path
		sentences = doc_to_sent(path, embeddings_lookup, model_args)
		push!(sentences_data, sentences)
	end

	vcat(sentences_data..., cols=:union)

end

document_dataframe = make_document_dataframe(deserialization_paths, model_args)

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
