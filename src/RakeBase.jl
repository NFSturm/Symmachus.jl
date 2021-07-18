using StatsBase
using IterTools
using WordTokenizers: tokenize
using MLStyle.Modules.Cond
using Chain
using DataStructures
using Transducers
using Combinatorics

sentence = "O Governo vem aqui pedir uma autorização legislativa para estabelecer o Sistema de Gestão Integrada de Fogos Rurais, mas, antes de apresentar esta autorização legislativa, cuidou de garantir a sua viabilização por parte do PSD."

tokenized_sentence = lowercase.(tokenize(sentence))::Vector{String}

global_sentence_lookup = Dict(1:length(tokenized_sentence) .=> tokenized_sentence)::Dict{Int64, String}

stopwords = open("./data/stopwords/pt_stopwords.txt") do file
    readlines(file)
end

no_stopword(word::String) = word ∈ stopwords ? true : false

words_integer = 1:1:length(tokenized_sentence) |> collect

function stopword_or_word(word_position::Int64, stopwords::Vector{String}, global_sentence_lookup::Dict{Int64, String})::Union{Nothing, Int64}
    string_word = getindex(global_sentence_lookup, word_position)
    result = no_stopword(string_word)
    return @cond begin
        result => nothing
        !result => word_position
    end
end


filtered_words = stopword_or_word.(words_integer, Ref(stopwords), Ref(global_sentence_lookup)) |> Filter(x -> !isnothing(x)) |> collect

#word_dict = Dict(1:length(Set(words)) .=> Set(words))::Dict{Int64, String}
#reverse_word_dict = Dict(Set(words) .=> 1:length(Set(words)))::Dict{String, Int64}

#words_integer = getindex.(Ref(reverse_word_dict), words)::Vector{Int64}

#word_frequencies = countmap(words_integer)::Dict{Int64, Int64}


function make_ngrams(words::Vector{Int64}, ngram_size::Int64)::Vector{NTuple{ngram_size, Int64}}
    collect(partition(words, ngram_size, 1))
end

ngrams = make_ngrams(filtered_words, 3)

ngram = (31, 34, 38)

max_distance = 2

constituent_pairs = partition(ngram, 2, 1) |> collect

constituents_close = @chain constituent_pairs begin
    reduce.(-, _)
    abs.(_)
    @cond begin
        any(x -> x > max_distance, _) => false
        all(x -> x < max_distance, _) => true
    end
end


function get_close_ngrams(ngram::NTuple{N, Int64}, max_distance::Int64)::Union{Nothing, NTuple{N, Int64}} where N
    constituent_pairs = partition(ngram, 2, 1) |> collect
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

filtered_ngrams = get_close_ngrams.(ngrams, 2) |> Filter(x -> !isnothing(x)) |> collect

node_lookup = Dict(1:length(filtered_words) .=> Set(filtered_words))

# Translate ngrams to axis length dictionary

partition_edgelist_2d(ngram::NTuple{N, Int64}) where N = combinations(ngram) |> collect |> Filter(x -> length(x) == 2) |> collect

edgelist = reduce(vcat, partition_edgelist_2d.(filtered_ngrams))


function fill_coocurrence_matrix!(edgelist::Vector{Vector{Int64}}, zero_matrix::Matrix{Float64})::Matrix{Float64}
    for edge in edgelist
        zero_matrix[edge...] += 1
        zero_matrix[reverse(edge)...] += 1
    end
    return zero_matrix
end

cooc_matrix = fill_coocurrence_matrix!(edgelist, zeros(length(Set(words)), length(Set(words))))::Matrix{Float64}

cooc_counts = Dict(1:size(cooc_matrix)[1] .=> vec(Int.(sum(cooc_matrix, dims=1))))::Dict{Int64, Int64}

function generate_word_scores(coocurrence::Dict{Int64, Int64}, word_frequencies::Dict{Int64, Int64})::Dict{Int64, Float64}
    word_scores = Dict()

    for key in keys(word_frequencies)
        freq = word_frequencies[key]::Int64
        cooc = cooc_counts[key]::Int64
        word_scores[key] = cooc/freq
    end
    return word_scores
end

word_scores = generate_word_scores(cooc_counts, word_frequencies)::Dict{Int64, Float64}

unique_ngrams = Set(filtered_ngrams)::Set{NTuple{N, Int64}} where N

function score_candiates(ngram::NTuple{N, Int64}, word_scores::Dict{Int64, Float64})::Tuple{NTuple{N, Int64}, Float64} where N
    ngram, reduce(+, getindex.(Ref(word_scores), ngram))
end

candidate_scores = score_candiates.(unique_ngrams, Ref(word_scores))::Vector{Tuple{NTuple{N, Int64}, Float64}} where N

function get_keyword_scores(candidate_score::Tuple{NTuple{N, Int64}, Float64}, word_dict::Dict{Int64, String}) where N
    lookup_words = candidate_score[1]
    words = getindex.(Ref(word_dict), lookup_words)
    return words, candidate_score[2]
end

scored_keywords = get_keyword_scores.(candidate_scores, Ref(word_dict))
