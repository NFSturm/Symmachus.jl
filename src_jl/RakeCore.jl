module RakeCore

using StatsBase
using IterTools
using WordTokenizers: tokenize
using MLStyle.Modules.Cond
using Chain
using Transducers
using Combinatorics
using Pipe

export rake, rake_wrapper, read_stopwords

function tokenize_sentence(sentence::String)::Vector{String}
    lowercase.(tokenize(sentence))
end

@doc """
    make_global_lookup(tokenized_sentence::Vector{String})::Dict{Int64, String}

Creates a global lookup dictionary based on the sentence position values in `tokenized_sentence`.
"""
function make_global_lookup(tokenized_sentence::Vector{String})::Dict{Int64, String}
    Dict(1:length(tokenized_sentence) .=> tokenized_sentence)::Dict{Int64, String}
end

function read_stopwords(path::String)::Vector{String}
    stopwords = open(path) do file
        readlines(file)
    end
    return stopwords
end

no_stopword(word::String, stopwords::Vector{String})::Bool = word ∈ stopwords ? true : false

function encode_words(tokenized_sentence::Vector{String})::Vector{Int64}
    1:1:length(tokenized_sentence) |> collect
end

function stopword_or_word(word_position::Int64, stopwords::Vector{String}, global_sentence_lookup::Dict{Int64, String})::Union{Nothing, Int64}
    string_word = getindex(global_sentence_lookup, word_position)
    result = no_stopword(string_word, stopwords)
    return @cond begin
        result => nothing
        !result => word_position
    end
end

function make_ngrams(words::Vector{Int64}, ngram_size::Int64)::Vector{NTuple{ngram_size, Int64}}
    collect(partition(words, ngram_size))
end

function get_close_ngrams(ngram::NTuple{N, Int64}, max_distance::Int64)::Union{Nothing, NTuple{N, Int64}} where N
    constituent_pairs = partition(ngram, 2) |> collect
    constituents_close = @chain constituent_pairs begin
        reduce.(-, _)
        abs.(_)
        @cond begin
            any(x -> x > max_distance, _) => false
            all(x -> x <= max_distance, _) => true
        end
    end
    if !constituents_close
        return nothing
    else
        return ngram
    end
end

get_word_from_index(lookup::Dict{Int64, String}, ngram::NTuple{N, Int64}) where N = getindex.(Ref(lookup), ngram)

partition_edgelist_2d(ngram::NTuple{N, Int64}) where N = combinations(ngram) |> collect |> Filter(x -> length(x) == 2) |> collect

@doc """
    fill_coocurrence_matrix!(edgelist::Vector{Vector{Int64}}, zero_matrix::Matrix{Float64})::Matrix{Float64}

Fills a coocurrence matrix based on a 2D-`edgelist`.
"""
function fill_coocurrence_matrix!(edgelist::Vector{Vector{Int64}}, zero_matrix::Matrix{Float64})::Matrix{Float64}
    for edge in edgelist
        zero_matrix[edge...] += 1
        zero_matrix[reverse(edge)...] += 1
    end
    return zero_matrix
end

function generate_word_scores(coocurrence::Dict{Int64, Int64}, word_frequencies::Dict{Int64, Int64})::Dict{Int64, Float64}
    word_scores = Dict()

    for key in keys(word_frequencies)
        freq = word_frequencies[key]::Int64
        cooc = coocurrence[key]::Int64
        word_scores[key] = cooc/freq
    end
    return word_scores
end

function score_candiates(ngram::NTuple{N, Int64}, word_scores::Dict{Int64, Float64})::Tuple{NTuple{N, Int64}, Float64} where N
    ngram, reduce(+, getindex.(Ref(word_scores), ngram))
end

function get_keyword_scores(candidate_score::Tuple{NTuple{N, Int64}, Float64}, word_dict::Dict{Int64, String}) where N
    lookup_words = candidate_score[1]
    words = getindex.(Ref(word_dict), lookup_words)
    return words, candidate_score[2]
end

make_ngrams_from_filtered_lookup(filtered_lookup::Dict{String, Int64}, string_ngram::NTuple{N, String}) where N = getindex.(Ref(filtered_lookup), string_ngram)

function rake(sentence::String, keyword_length::Int64, stopwords::Vector{String})::Vector{Tuple{NTuple{N, String} where N, Float64}}
    tokenized_sentence = tokenize_sentence(sentence)
    lookup = make_global_lookup(tokenized_sentence)
    words_encoded = encode_words(tokenized_sentence)

    # Removing stopwords as defined
    filtered_words = stopword_or_word.(words_encoded, Ref(stopwords), Ref(lookup)) |> Filter(x -> !isnothing(x)) |> collect

    filtered_words = stopword_or_word.(words_encoded, Ref(stopwords), Ref(lookup)) |> Filter(x -> !isnothing(x)) |> collect
    ngrams = make_ngrams(filtered_words, keyword_length)

    string_ngrams = get_word_from_index.(Ref(lookup), ngrams)

    filtered_word_strings = getindex.(Ref(lookup), filtered_words)

    filtered_lookup = Dict(filtered_word_strings .=> 1:length(filtered_word_strings))
    filtered_lookup_reverse = Dict(value => key for (key, value) in filtered_lookup)

    filtered_words_count = countmap(getindex.(Ref(filtered_lookup), filtered_word_strings))

    filtered_ngrams = make_ngrams_from_filtered_lookup.(Ref(filtered_lookup), string_ngrams)

    edgelist = reduce(vcat, partition_edgelist_2d.(filtered_ngrams))

    cooc_matrix = fill_coocurrence_matrix!(edgelist, zeros(length(Set(words_encoded)), length(Set(words_encoded))))::Matrix{Float64}

    cooc_counts = Dict(1:size(cooc_matrix)[1] .=> vec(Int.(sum(cooc_matrix, dims=1))))::Dict{Int64, Int64}
    word_scores = generate_word_scores(cooc_counts, filtered_words_count)::Dict{Int64, Float64}

    unique_ngrams = Set(filtered_ngrams)::Set{NTuple{N, Int64}} where N
    candidate_scores = score_candiates.(unique_ngrams, Ref(word_scores))::Vector{Tuple{NTuple{N, Int64}, Float64}} where N

    scored_keywords = get_keyword_scores.(candidate_scores, Ref(filtered_lookup_reverse))

    num_keywords = length(Set(words_encoded))/3 |> floor |> Int

    top_keywords = sort(scored_keywords, by = x -> x[2], rev=true)[1:min(length(scored_keywords), num_keywords)]

    return top_keywords
end

@doc """
    rake_wrapper(text::String, keyword_length::Int64, stopwords::Vector{String})

A wrapper around *rake* to handle exceptions.
"""
function rake_wrapper(text::String, keyword_length::Int64, stopwords::Vector{String})
    try
        rake(text, keyword_length, stopwords)
    catch
        []
    end
end

end
