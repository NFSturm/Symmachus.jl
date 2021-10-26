using Serialization
using DataFrames
using Distributed
using UnPack
using MLStyle.Modules.Cond
using StatsBase
using Revise
using CSV

addprocs(4)

@everywhere begin
    using Pkg; Pkg.activate(".")

    using Serialization
    using DataFrames
    using Distributed
    using UnPack
    using MLStyle.Modules.Cond
    using StatsBase
    using CSV
end

@everywhere include("SymmachusCore.jl")
@everywhere using .SymmachusCore

@everywhere begin

    embeddings_vocab, embeddings = load_fasttext_embeddings("./data/embeddings")

    const word_lookup_table = get_word_lookup_table(embeddings_vocab)

    global embeddings_lookup = Lookup(word_lookup_table, embeddings, embeddings_vocab)

    mutable struct Args
        max_discourse_context::Int64
        max_sentence_context::Int64
        self_weight::Float64
    end
end

@everywhere function make_dataframe_row(sentence::Sentence, embedded_sentence::Vector{Float64})
    @unpack doc_uuid, sentence_id, sentence_text, actor_name, discourse_time = sentence
    return doc_uuid, sentence_id, sentence_text, actor_name, discourse_time, embedded_sentence
end


@everywhere @doc """
    doc_to_sent(file::String)

Using the `embeddings_lookup`, document files are read from the directory and \n
and embedded. The output is a dataframe containing the individual sentences. \n
`Args` defines a parameter struct.
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


@doc """
    make_sentence_dataframe(dir::String, args)

Embeds all documents in a directory and splits them into sentences. Returns \n
the complete dataframes with sentences.
"""``
function make_sentence_dataframe(dir::String, args)``

    # Reading speech docs from disk
    files = readdir(dir, join=true)[1:50]

    # Mapping `doc_to_sent` to the files.
    res = pmap(doc_to_sent, files)

    return res
end


@doc """
    concat_dataframes(dataframes::Vector{DataFrame})::DataFrame

A simple function wrapper around `vcat`.
"""
function concat_dataframes(dataframes::Vector{DataFrame})::DataFrame
    vcat(dataframes..., cols=:union)
end

@everywhere global args = Args(3, 3, 0.7)

sentence_dataframe = make_sentence_dataframe("./data/speech_docs", args) |> concat_dataframes

labelling_sentences = sentence_dataframe[!, [:doc_uuid, :sentence_id, :sentence_text, :actor_name, :discourse_time]]

labelling_sentences[!, :labels] = sample(0:1, nrow(labelling_sentences))

CSV.write("./data/example_docs.csv", labelling_sentences)
