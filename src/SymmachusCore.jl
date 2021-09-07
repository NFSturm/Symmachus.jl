module SymmachusCore

using Statistics
using Serialization: deserialize
using MLStyle.Modules.Cond
using Chain
using StringDistances
using Distributions: Beta

export Sentence,
    Document,
    Lookup,
    get_embedding_from_string,
    get_word_lookup_table,
    load_fasttext_embeddings,
    get_nearest_word_substitute,
    substitute_word_with_nearest_neighbour,
    focus_scaler,
    generate_word_focus_distribution,
    generate_syntactic_window,
    fill_dependency_matrix!,
    make_dependency_matrix,
    sample_dependency_edges,
    get_axis_embeddings,
    embed_sentence,
    get_context_embedding_window,
    calculate_context_weights,
    weight_position,
    weight_embeddings,
    get_weighted_context_embeddings,
    embed_document

#************ LANGUAGE STRUCTS ************

struct Sentence
    lookup::Dict{Int64, String}
    dep_graph::Vector{Vector{Int64}}
    root::Int64
    doc_length::Int64
    sentence_text::String
    doc_uuid::String
    sentence_id::Int64
    actor_name::String
    discourse_time::String
end

# A simple container for the sentences in a document
struct Document
    sentences::Vector{Sentence}
end

struct Lookup
    word_lookup_table::Dict{String, Int64}
    embeddings::Matrix{Float32}
    embeddings_vocab::Vector{String}
end

#************ EMBEDDING UTILITIES ************

@doc """
    get_word_lookup_table(embeddings_vocab::Vector{String})::Dict{String, Int64}

Using the `embeddings_vocab`, each word is mapped to an index.
"""
function get_word_lookup_table(embeddings_vocab::Vector{String})::Dict{String, Int64}
    return Dict(word=>ii for (ii,word) in enumerate(embeddings_vocab))
end


@doc """
Returns a 300-element (FastText) embedding for a given string.
"""
function get_embedding_from_string(word::String, lookup::Dict{String, Int64}, embeddings_table::Matrix{Float32})
    index = lookup[word]
    embedding = embeddings_table[:,index]
    return embedding
end

@doc """
    load_fasttext_embeddings(data_dir::String)

Loads and deserializes FastText embeddings from `data_dir`.
"""
function load_fasttext_embeddings(data_dir::String)
    embeddings_vocab = deserialize("$(data_dir)/fasttext_pt_embeddings_vocab.jls")
    embeddings = deserialize("$(data_dir)/fasttext_pt_embeddings.jls")
    return embeddings_vocab, embeddings
end

#************ STRING SUBSTITUTION ************

function get_nearest_word_substitute(word::String, word_lookup::Dict{String, Int64})::String
    return findnearest(word, collect(keys(word_lookup)), RatcliffObershelp())[1]
end

function substitute_word_with_nearest_neighbour(word::String, word_lookup_table::Dict{String, Int64})
    substitute_word = @cond begin
        word ∈ keys(word_lookup_table) => word
        word ∉ keys(word_lookup_table) => get_nearest_word_substitute(word, word_lookup_table)
    end
    return substitute_word
end

#************ WORD FOCUS ************

@doc """
    focus_scaler(sentence_root_index::Int64, sentence_length::Int64)::Int64

This function generates a scale adjustment to stretch the focus distribution to
prevent too tight sampling densities. It is calculated as the `sentence_length`
divided by the index of the `sentence_root`. Finally, we multiply this value by two.
"""
function focus_scaler(sentence_root_index::Int64, sentence_length::Int64)::Int64
    return Int(ceil(sentence_length/sentence_root_index)*2)
end

@doc """
    generate_word_focus_distribution(sentence_root_index::Int64, sentence_length::Int64)

This is the base mechanism to emulate **attention** in a lightweight way. The main ingredient is a scaled \n
and shifted Beta distribution, from which a single sample is taken. Said distribution is scaled in such a \n
way that words closer to the sentence root a sampled more frequently, thus exposing the assumption that \n
the sentence is root is that part of a sentence that captures the reader's attention more easily to di- \n
gest the information contained in the sentence.

This mechanism needs the `sentence_root_index` as well as the overall `sentence_length`.
"""
function generate_word_focus_distribution(sentence_root_index::Int64, sentence_length::Int64)
    scale_adjustment = focus_scaler(sentence_root_index, sentence_length)::Union{Float64, Int64}
    dist = 1 + sentence_length*Beta(sentence_root_index + scale_adjustment, sentence_length)
    return dist
end


@doc """
    generate_syntactic_window(token_index::Int64, doc_length::Int64, window_pad::Int64)::Vector{Int64}

Given a `token_index` and `sentence_length` a syntactic window, comprising the indices near the `token_index`, is generated. \n
The width of the syntanctic window can be changed by setting `window_pad`.
"""
function generate_syntactic_window(token_index::Int64, sentence_length::Int64, window_pad::Int64)::Vector{Int64}
    res = @cond begin
        token_index - window_pad >= 1 && token_index + window_pad <= sentence_length => token_index - window_pad:1:token_index + window_pad
        token_index - window_pad < 1 && token_index + window_pad <= sentence_length => 1:1:token_index + window_pad
        token_index - window_pad >= 1 && token_index + window_pad > sentence_length => token_index - window_pad:1:sentence_length
        token_index - window_pad < 1 && token_index + window_pad > sentence_length => 1:1:sentence_length
    end
    return res
end

#************ SENTENCE DEPENDENCY GRAPH UTILITIES ************

@doc """
    fill_dependency_matrix!(edgelist::Vector{Vector{Int64}}, zero_matrix::Matrix{Float64})

Given an `edgelist`, an initial `zero_matrix` is filled.
"""
function fill_dependency_matrix!(edgelist::Vector{Vector{Int64}}, zero_matrix::Matrix{Float64})::Matrix{Float64}
    for edge in edgelist
        zero_matrix[edge...] = 1
    end
    return zero_matrix
end

@doc """
    make_dependency_matrix(sentence::Sentence)

Given a `Sentence` struct, a dependency matrix is generated.
"""
function make_dependency_matrix(sentence::Sentence)::Matrix{Float64}
    edgelist = sentence.dep_graph::Vector{Vector{Int64}}
    doc_length = sentence.doc_length::Int64
    sentence_adjacency_matrix = fill_dependency_matrix!(edgelist, zeros(doc_length, doc_length))
    return sentence_adjacency_matrix
end


@doc """
    sample_dependency_edges(sentence_root::Int64, sentence_length::Int64, window_padding::Int64)

Given the `sentence_root`of a sentence and a corresponding `sentence_length`, samples from rows and columns are taken \n
while focussing the "attention" of the sampling on the indices closer to the sentence root. The width of the window thus derived \n
can be changed by setting `window_padding`.
"""
function sample_dependency_edges(sentence_root::Int64, sentence_length::Int64, window_padding::Int64)::Vector{Int64}
    scale_adjustment = Int(focus_scaler(sentence_root, sentence_length))::Int64
    sentence_dist = generate_word_focus_distribution(sentence_root, sentence_length)
    selector = Int(ceil(getindex(rand(sentence_dist, 1), 1)))::Int64
    syntactic_window = generate_syntactic_window(selector, sentence_length, window_padding)::Vector{Int64}
    return syntactic_window
end


@doc """
    get_axis_embeddings(axis_embedding, axis_indices, sentence_lookup, lookup)

Returns axis embeddings given an `axis_mapping`, `axis_indices` as well as `sentence_lookup` \n
and the document.
"""
function get_axis_embeddings(axis_mapping::Dict{Int64, Int64}, axis_indices::Vector{Int64}, sentence_lookup::Dict{Int64, String}, lookup::Lookup)::Vector{Vector{Float32}}
    axis_embeddings = @chain axis_mapping begin
        getindex.(Ref(_), axis_indices)::Vector{Int64}
        getindex.(Ref(sentence_lookup), _)::Vector{String}
        substitute_word_with_nearest_neighbour.(_, Ref(lookup.word_lookup_table))::Vector{String}
        get_embedding_from_string.(_, Ref(lookup.word_lookup_table), Ref(lookup.embeddings))::Vector{Vector{Float32}}
    end
    return axis_embeddings
end

@doc """
    embed_sentence(sentence::Sentence)::Vector{Float32}

This function performs the word focus embedding by using a `sentence` of type Sentence.
"""
function embed_sentence(sentence::Sentence, embeddings_lookup::Lookup, sentence_context_size::Int64)::Vector{Float32}
    sentence_lookup = sentence.lookup  # Deconstructing the sentence

    # Matrix should be symmetric
    sent_dep_matrix = make_dependency_matrix(sentence)

    function get_subset_matrix(sentence::Sentence, sentence_context_size::Int64)
        subset_null = true::Bool
        while subset_null
            # Rows and columns from the sentence dependency matrix are sampled
            row_samples = sample_dependency_edges(sentence.root, sentence.doc_length, sentence_context_size)::Vector{Int64}

            subset_matrix = sent_dep_matrix[row_samples, :]::Matrix{Float64}

            if sum(subset_matrix) == 0
                continue
            else
                subset_null = false::Bool
                return subset_matrix, row_samples
            end
        end
    end

    subset_matrix, row_samples = get_subset_matrix(sentence, sentence_context_size)

    # From the subset, only those entries are extracted that are non-null, i.e.
    syntactic_flow_indices = findall(!iszero, subset_matrix)

    # Mapping inner row/column indices to those of the outer matrix
    row_mapping = Dict(1:size(subset_matrix)[1] .=> row_samples)

    row_indices = [index[1] for index in syntactic_flow_indices]::Vector{Int64}
    column_indices = [index[2] for index in syntactic_flow_indices]::Vector{Int64}

    row_word_embeddings = get_axis_embeddings(row_mapping, row_indices, sentence_lookup, embeddings_lookup)

    column_word_embeddings = @chain column_indices begin
        getindex.(Ref(sentence_lookup), _)::Vector{String}
        substitute_word_with_nearest_neighbour.(_, Ref(embeddings_lookup.word_lookup_table))::Vector{String}
        get_embedding_from_string.(_, Ref(embeddings_lookup.word_lookup_table), Ref(embeddings_lookup.embeddings))::Vector{Vector{Float32}}
    end

    # Columns (sentence children) are subtracted from row embeddings (heads)
    semantic_space_walk = mean([row_word_embeddings[i] - column_word_embeddings[i] for i in 1:length(row_word_embeddings)])::Vector{Float32}

    return semantic_space_walk
end

#************ DOCUMENT CONTEXT EMBEDDINGS ************
@doc """
    get_context_embedding_window(embedded_sentences::Vector{Vector{Float32}}, discourse_context_size::Int64, discourse_length::Int64)

`embedded_sentences` are those sentences that have gone through the SymmachusCore embedding. The `discourse_context_size` is the maximum context that should \n
be considered for the final embedding. `discourse_length` refers to the total length of the document.
"""
function get_context_embedding_window(embedded_sentences::Vector{Vector{Float32}}, discourse_context_size::Int64, discourse_length::Int64)

    context_embedding_positions = []

    for sentence_pos in 1:length(embedded_sentences)
        context_embedding_indices = @cond begin
            sentence_pos - discourse_context_size >= 1 && sentence_pos + discourse_context_size <= discourse_length => sentence_pos - discourse_context_size:1:sentence_pos + discourse_context_size
            sentence_pos - discourse_context_size < 1 && sentence_pos + discourse_context_size <= discourse_length => 1:1:sentence_pos + discourse_context_size
            sentence_pos - discourse_context_size >= 1 && sentence_pos + discourse_context_size > discourse_length => sentence_pos - discourse_context_size:1:discourse_length
            sentence_pos - discourse_context_size < 1 && sentence_pos + discourse_context_size > discourse_length => 1:1:discourse_length
        end
        push!(context_embedding_positions, (sentence_pos, collect(context_embedding_indices)))
    end

    return context_embedding_positions
end

@doc """
    calculate_context_weights(self_weight::Float64, context_size::Int64)::Float64

Given a `self_weight`, the weights for the remaining elements in the context (for `context_size`) are calculated. \n
From the weight, 1 is subtracted to account for the factor, that the current element has weight `self_weight`.
"""
function calculate_context_weights(self_weight::Float64, context_size::Int64)::Float64
    (1 - self_weight)/(context_size-1)
end

@doc """
    weight_position(weight_dict::Dict{String, Float64}, self_position::Int64, position::Int64)

Given a `weight_dict, the condition whether the `position` argument is the `self_position` argument. If it is, the \n
self-weight is returned, otherwise the remainder weight.
"""
weight_position(weight_dict::Dict{String, Float64}, self_position::Int64, position::Int64) = position == self_position ? weight_dict["self"] : weight_dict["context"]

@doc """
    weight_embeddings(embeddings::Vector{Vector{Float32}}, weights::Vector{Float64})::Vector{Float64}

Given a vector of `embeddings` and corresponding positional `weights`, an averaged embedding is returned.
"""
function weight_embeddings(embeddings::Vector{Vector{Float32}}, weights::Vector{Float64})::Vector{Float64}
    reduce(+, [weights[i] .* embeddings[i] for i in 1:length(embeddings)])
end

@doc """
    get_weighted_context_embeddings(embedded_sentences::Vector{Vector{Float32}}, self_weight::Float64, discourse_context_size::Int64)

This function calculates context embeddings for each sentence in `embedded_sentences`. Arguments that need to be set are the `self_weight` \n
and the `discourse_context_size`.
"""
function get_weighted_context_embeddings(embedded_sentences::Vector{Vector{Float32}}, self_weight::Float64, discourse_context_size::Int64)

    context_embedding_windows = get_context_embedding_window(embedded_sentences, discourse_context_size, length(embedded_sentences))

    weighted_context_embeddings = Vector{Float64}[]

    for i in 1:length(embedded_sentences)
        self_position, positions = context_embedding_windows[i]
        context_weights = calculate_context_weights(self_weight, length(positions))
        weight_dict = Dict("self" => self_weight, "context" => context_weights)
        positional_weights = weight_position.(Ref(weight_dict), Ref(self_position), positions)
        weighted_context_embedding = weight_embeddings(embedded_sentences[positions], positional_weights)
        push!(weighted_context_embeddings, weighted_context_embedding)
    end
    return weighted_context_embeddings
end

@doc """
    embed_document(document::Document, discourse_context_size::Int64, sentence_context_size::Int64, self_weight::Float64, embeddings_lookup::Lookup)::Vector{Vector{Float64}}

This generates a **Symmachus** embedding. A `Document` consisting of sentences is used. `discourse_context_size`and `sentence_context_size` need to be specified. The `self_weight`refers \n
to how much the embedding of the individual sentence should be weighted, as opposed to its neighbouring sentences. Finally, the `embeddings_lookup` refers to a lookup table for FastText embeddings.
"""
function embed_document(document::Document, discourse_context_size::Int64, sentence_context_size::Int64, self_weight::Float64, embeddings_lookup::Lookup)::Vector{Vector{Float64}}
    embedded_sentences = [embed_sentence(sentence, embeddings_lookup, sentence_context_size) for sentence in document.sentences]

    context_embedding_windows = get_context_embedding_window(embedded_sentences, discourse_context_size, length(embedded_sentences))

    return get_weighted_context_embeddings(embedded_sentences, self_weight, discourse_context_size)
end

end
