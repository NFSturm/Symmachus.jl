using Statistics
using Serialization: deserialize
using MLStyle.Modules.Cond
using Chain

include("WordFocus.jl")
include("LanguageStructs.jl")
include("EmbeddingAccess.jl")
include("StringMatcher.jl")

using .LanguageStructs
using .WordFocus
using .EmbeddingAccess
using .StringMatcher

embeddings_vocab, embeddings = load_fasttext_embeddings("./data/embeddings")

const word_lookup_table = get_word_lookup_table(embeddings_vocab)

embeddings_lookup = Lookup(word_lookup_table, embeddings, embeddings_vocab)

Doc = deserialize("./data/speech_docs/018edf9b-99de-4a96-b137-773d01a95f75.jls")

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

sentence = Doc.sentences[4]

sentence_lookup = sentence.lookup

sent_dep_matrix = make_dependency_matrix(sentence)

@doc """
    sample_dependency_edges(sentence_root::Int64, sentence_length::Int64, window_padding::Int64)

Given the `sentence_root`of a sentence and a corresponding `sentence_length`, samples from rows and columns are taken \n
while focussing the "attention" of the sampling on the indices closer to the sentence root. The width of the window thus derived \n
can be changed by setting `window_padding`.
"""
function sample_dependency_edges(sentence_root::Int64, sentence_length::Int64, window_padding::Int64)::Vector{Int64}
    scale_adjustment = Int(focus_scaler(sentence.root, sentence.doc_length))::Int64
    sentence_dist = generate_word_focus_distribution(sentence.root, sentence.doc_length)
    selector = Int(ceil(getindex(rand(sentence_dist, 1), 1)))::Int64
    syntactic_window = generate_syntactic_window(selector, sentence.doc_length, window_padding)::Vector{Int64}
    return syntactic_window
end

row_samples = sample_dependency_edges(sentence.root, sentence.doc_length, 5)::Vector{Int64}
column_samples = sample_dependency_edges(sentence.root, sentence.doc_length, 5)::Vector{Int64}

subset_matrix = sent_dep_matrix[row_samples, column_samples]

row_mapping = Dict(1:size(subset_matrix)[1] .=> row_samples)
column_mapping = Dict(1:size(subset_matrix)[1] .=> column_samples)

syntactic_flow_indices = findall(!iszero, subset_matrix)

row_indices = [index[1] for index in syntactic_flow_indices]::Vector{Int64}
column_indices = [index[2] for index in syntactic_flow_indices]::Vector{Int64}

@doc """
    get_axis_embeddings(axis_embedding, axis_indices, sentence_lookup, lookup)

Returns axis embeddings given an `axis_mapping`, `axis_indices` as well as `sentence_lookup` \n
and the document.
"""
function get_axis_embeddings(axis_mapping::Dict{Int64, Int64}, axis_indices::Vector{Int64}, sentence_lookup::Dict{Int64, String}, document_lookup::Lookup)::Vector{Vector{Float32}}
    axis_embeddings = @chain axis_mapping begin
        getindex.(Ref(_), axis_indices)::Vector{Int64}
        getindex.(Ref(sentence_lookup), _)::Vector{String}
        substitute_word_with_nearest_neighbour.(_, Ref(document_lookup.word_lookup_table))::Vector{String}
        get_embedding_from_string.(_, Ref(document_lookup.word_lookup_table), Ref(document_lookup.embeddings))::Vector{Vector{Float32}}
    end
    return axis_embeddings
end

row_word_embeddings = get_axis_embeddings(row_mapping, row_indices, sentence_lookup, embeddings_lookup)
column_word_embeddings = get_axis_embeddings(column_mapping, column_indices, sentence_lookup, embeddings_lookup)

semantic_space_walk = mean(row_word_embeddings .- column_word_embeddings)::Vector{Float32}
