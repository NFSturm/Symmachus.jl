# Self-Training

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

	# Initializing the Symmachus model grid
	grid = Dict(:max_discourse_context_size => 1:10, :max_sentence_context_size => 1:15, :self_weight => 0.5:0.05:0.9, :grid_size => 10)

	symmachus_args_array = [SymmachusArgs(
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

	# Initializing the boosting grid
	boosting_grid = Dict(
		:num_rounds => 50:15:165,
		:params => ["max_depth" => 2:5, "eta" => 0.1:0.1:1],
		:true_threshold => 0.5:0.05:0.7,
		:grid_size => 10
	)

	boosting_args_array = [BoostingArgs(
		num_rounds = sample(boosting_grid[:num_rounds], 1) |> first,
		metrics=["aucpr"],
		params = [
		"max_depth" => sample(boosting_grid[:params][1][2], 1) |> first |> Int,
		"eta" => sample(boosting_grid[:params][2][2], 1) |> first,
		"objective" => "binary:logistic"
		],
		true_threshold = sample(boosting_grid[:true_threshold], 1) |> first,
		train_prop = 0.8
	) for i in 1:boosting_grid[:grid_size]]

end

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

#@everywhere sentence_label_data = deserialize("./cache/sentence_label_data.jls")

@doc """
    get_labelled_sentences_from_documents(document::Tuple{DataFrame, SymmachusArgs}, labelled_sentences::DataFrame)::DataFrame

Extracts those sentences from embedded `documents` that are in `labelled_sentences` \n
by performing an inner join.
"""
function get_labelled_sentences_from_documents(document::Tuple{DataFrame, SymmachusArgs}, labelled_sentences::DataFrame)::Tuple{DataFrame, SymmachusArgs}
	document_embedded, symmachus_args = document
	select!(document_embedded, Not([:actor_name, :sentence_text])) # Avoiding duplicate columns downstream
	labelled_sentences = innerjoin(labelled_sentences, document_embedded, on=[:doc_uuid, :sentence_id], makeunique=true)
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
    boost(sentence_label_data::Tuple{DataFrame, SymmachusArgs}, boosting_args_array::Vector{BoostingArgs})
Trains a booster on `feature_data` and `label_data`. Arguments of the model can be \n
specified using `boosting_args`.
"""
function boost(sentence_label_data::Tuple{DataFrame, SymmachusArgs}, boosting_args_array::Vector{BoostingArgs})
	sentence_label_data, symmachus_args = sentence_label_data
	X_train, y_train, X_test, y_test = make_train_test_data(sentence_label_data, "sentence_embedding", "label", first(boosting_args_array).train_prop)

	model_uuid = uuid4() |> string

	boosters = []

	for arg in boosting_args_array

		@suppress begin
			bst = train_booster(X_train, y_train, arg)
			predictions = predict_booster(bst, X_test)
			confmat = confusion_matrix(predictions, y_test, arg.true_threshold)
			f1_model_score = f1_score(confmat)
			push!(boosters, Dict(
				:model_id => model_uuid,
				:f1_score => f1_model_score,
				:model_args => arg,
				:symmachus_args => symmachus_args
				)
			)
		end
	end

	best_booster_model = @chain boosters begin
		sort(_, by=x -> x[:f1_score], rev=true)
		first
	end

	sentence_label_data, best_booster_model
end


@doc """
    boost_sentence_data(sentence_label_data::Vector{DataFrame}, args::Vector{BoostingArgs})

Trains an XGBoost model in parallel on `sentence_label_data` and `args`, which \n
are the boosting args.
"""
function boost_sentence_data(sentence_label_data::Vector{Tuple{DataFrame, SymmachusArgs}}, args::Vector{BoostingArgs})
	res = pmap((s, arg) -> boost(s, arg), sentence_label_data, Iterators.repeated(args, length(sentence_label_data)) |> collect)
end

@doc """
    select_best_model(symmachus_boost_models::Vector{Dict{Symbol, Any}}, scoring_metric::String)

Selects the best model with specifications based on the `scoring_metric` used.
"""
function select_best_model(symmachus_boost_models::Vector{Tuple{DataFrame, Dict{Symbol, Any}}}, scoring_metric::String)
	@chain symmachus_boost_models begin
		sort(_, by=x -> last(x)[Symbol(scoring_metric)], rev=true)
		first
	end
end


@doc """
    sample_documents(all_documents_path::String, labelled_documents::Vector{String}, num_documents::Int64)::Vector{String}
Samples documents from a directory. Returns a vector of strings.
"""
function sample_documents(all_documents_path::String, labelled_sentences::DataFrame, num_documents::Int64)::Vector{String}
	all_documents = readdir(all_documents_path) # The path to the documents
	all_documents_id = first.(split.(all_documents, Ref('.')))

	uuid_labels = labelled_sentences[!, :doc_uuid] |> Set |> collect

	labels = labelled_sentences[!, :doc_uuid]

	existing_labels = labelled_sentences[!, :doc_uuid] |> Set |> collect

	document_samples = sample(all_documents_id, num_documents)

	# Sample only those documents that have not yet been labelled.
	zips = zip(document_samples, document_samples .∈ Ref(existing_labels)) |> collect

	[first(zip) for zip in zips if !last(zip)]
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

	feature_labels_sorted_rev = sort(broadcast_targets, :label, rev=true)
	feature_labels_sorted = sort(broadcast_targets, :label)

	confident_predictions_rev = feature_labels_sorted_rev[1:75, :]
	confident_predictions = feature_labels_sorted[1:75, :]

	confident_predictions_all = concat_dataframes([confident_predictions_rev, confident_predictions])

	transform(confident_predictions_all, [:label] .=> ByRow(x -> round(x) |> Int) .=> [:label])

end


@doc """
    cache_model(label_data::DataFrame, model_specs::Dict{Symbol, Any}, cache_path::String)

Caches a model characterized by `label_data` and `model_specs`.
"""
function cache_model(label_data::DataFrame, model_specs::Dict{Symbol, Any}, cache_path::String)
	# Creating cache info
	cache_info_date = @pipe today() |> string |> replace(_, "-" => "")
	cache_info_time = @pipe now() |> Time(_) |> string |> replace(_, ":" => "") |> split(_, ".") |> first

	cache_info = join([cache_info_date, cache_info_time], "_")

	@info "Cacheing label data"
	serialize(joinpath(cache_path, "label_data_" * cache_info * ".jls"), label_data)

	@info "Cacheing model specifications"
	serialize(joinpath(cache_path, "model_specs_" * cache_info * ".jls"), model_specs)
end



@doc """
    sample_mutants(seed_argument::SymmachusArgs, sentence_context_spectrum::Int64, discourse_context_spectrum::Int64, self_weight_spectrum::Float64)::Vector{SymmachusArgs}

Using a `seed_argument`, a vector of new *SymmachusArgs* is generated. The mutation can be tuned used `sentence_context_spectrum`, \n
`discourse_context_spectrum` and `self_weight_spectrum`. Returns the array of mutated *SymmachusArgs*.
"""
function sample_mutants(seed_argument::SymmachusArgs, sentence_context_spectrum::Int64, discourse_context_spectrum::Int64, self_weight_spectrum::Float64)::Vector{SymmachusArgs}
    @unpack max_discourse_context_size, max_sentence_context_size, self_weight = seed_argument

    mutant_grid = Dict(
                    :max_discourse_context_size => (max_discourse_context_size-discourse_context_spectrum):(max_discourse_context_size+discourse_context_spectrum),
                    :max_sentence_context_size => (max_sentence_context_size-sentence_context_spectrum):(max_sentence_context_size+sentence_context_spectrum),
                    :self_weight => (self_weight-self_weight_spectrum):0.05:(self_weight+self_weight_spectrum),
                    :grid_size => 10
    )

    function validate_argument(arg::Float64)
        res = @cond begin
            arg > 1 => 1
            arg < 0 => 0
            _ => arg
        end
    end

    validate_argument(arg::Int64) = arg < 0 ? 0 : arg

    symmachus_mutant_array = [SymmachusArgs(
        max_discourse_context_size = sample(mutant_grid[:max_discourse_context_size], 1) |> first |> validate_argument,
        max_sentence_context_size = sample(mutant_grid[:max_sentence_context_size], 1) |> first |> validate_argument,
        self_weight = sample(mutant_grid[:self_weight], 1) |> first |> validate_argument
    ) for i in 1:mutant_grid[:grid_size]]

    symmachus_mutant_array
end


@doc """
    train_self(labelled_data_path::String, unlabelled_data_path::String, symmachus_args_array::Vector{SymmachusArgs}, boosting_args_array::Vector{BoostingArgs}, iter_num::Int64, cache_path::String)

Initiates the self-training loop of an XGBoost model to label a set of initial binary labels, \n
contained in `labelled_data_path`. Gradually, new samples are drawn from `unlabelled_data_path`. \n
Hyperparameters for this model can be specified using `symmachus_args_array` for the sentence embedding model \n
and `boosting_args_array` for XGBoost params.
"""
function train_self(labelled_data_path::String, unlabelled_data_path::String, symmachus_args_array::Vector{SymmachusArgs}, boosting_args_array::Vector{BoostingArgs}, iter_num::Int64, cache_path::String)

	# Read labelled data from disk
	seed_label_data = DataFrame(CSV.File(labelled_data_path))

	# Read unlabelled file names from disk
	files = readdir(unlabelled_data_path)

	# Label data container
	label_data_container = DataFrame[]
	push!(label_data_container, seed_label_data)

	# Container for best model specifications
	model_specs_container = Dict[]

	for iter in 1:iter_num

		label_data = last(label_data_container)

		deserialization_items = collect(Set(label_data[!, :doc_uuid])) # Retrieves only unique docs

		deserialization_paths = make_deserialization_paths(deserialization_items)

		documents = retrieve_documents(deserialization_paths)

		if iter > 1 # For each iteration after the first one, the random mutation applies
			mutated_symmachus_args_array = sample_mutants(last(model_specs_container[:symmachus_args]), 2, 3, 0.2) # Can be parameterized

			# Create a separate embedding for each argument in symmachus_args_array
			sentence_label_dataframes = [(make_document_dataframe(documents, symmachus_arg) |> concat_dataframes, symmachus_arg) for symmachus_arg in mutated_symmachus_args_array]

		else # Only for first iteration to generate a seed struct
			# Create a separate embedding for each argument in symmachus_args_array
			sentence_label_dataframes = [(make_document_dataframe(documents, symmachus_arg) |> concat_dataframes, symmachus_arg) for symmachus_arg in symmachus_args_array]
		end

		sentence_label_data = get_labelled_sentences_from_documents.(sentence_label_dataframes, Ref(label_data))

		# Create the best boosting model for each dataframe
		symmachus_boost_models = boost_sentence_data(sentence_label_data, boosting_args_array)

		embedded_sentences, best_model = select_best_model(symmachus_boost_models, "f1_score")

		# Updating the model spec container
		push!(model_specs_container, Dict(:symmachus_args => best_model[:symmachus_args], :boosting_args => best_model[:model_args], :performance => best_model[:f1_score]))

		@info "Current iteration performs at: $(best_model[:f1_score])"

		new_deserialization_paths = sample_documents(unlabelled_data_path, embedded_sentences, 50) |> make_deserialization_paths

		new_documents = retrieve_documents(new_deserialization_paths) # Actually deserializes the documents

		new_document_dataframe = make_document_dataframe(
			new_documents, last(model_specs_container)[:symmachus_args]) |> concat_dataframes

		new_sentences = broadcast_labels(best_model, embedded_sentences, new_document_dataframe)

		new_data_union = concat_dataframes([embedded_sentences, new_sentences])

		# Removing the cached label DataFrame
		deleteat!(label_data_container, 1)

		# Updating the data container
		push!(label_data_container, new_data_union)

		@info "Dataframe was updated. Current length: $(nrow(new_data_union))"

		if iter % 5 == 0
			cache_model(new_data_union, best_model, cache_path)
		end

	end

	# Returning the final labelled data as well as the model training history
	last(label_data_container), model_specs_container

end

labelled_data_final, model_history = train_self("./data/labels/labels.csv", "./data/speech_docs", symmachus_args_array, boosting_args_array, 1, "./cache")
