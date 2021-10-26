using ThreadsX
using Serialization
using DataFrames
using UnPack
using DelimitedFiles

include("SymmachusCore.jl")

using .SymmachusCore

@doc """
    get_sentences(file::String)

Retrieves the individual sentences from a document. Takes a `file` as string and \n
returns a  containing the sentences.
"""
function get_sentences(file::String)

    sentences = DataFrame(
        doc_uuid = String[],
        sentence_id = Int64[],
        sentence_text = String[],
        actor_name = String[],
    )

    doc = deserialize(file)

    for sentence in doc.sentences
        @unpack doc_uuid, sentence_id, sentence_text, actor_name, discourse_time = sentence
        push!(sentences, [doc_uuid, sentence_id, sentence_text, actor_name])
    end

    return sentences
end

sents = get_sentences("./data/speech_docs/cfb0b4ec-3cbb-4c40-b6a3-31a4873ed9ed.jls")

sents

@doc """
    make_sentence_dataframe(directory::String)

Loops through a `directory` and retrieves all sentences from documents. Returns \n
an aggregated DataFrame with all sentence contained in the directory's documents.
"""
function make_sentence_dataframe(directory::String)
    sentence_dataframes = []

    ThreadsX.foreach(readdir(directory, join=true)) do f
        df = get_sentences(f)
        push!(sentence_dataframes, df)
    end

    return vcat(sentence_dataframes...)

end

labelling_sentences = make_sentence_dataframe("./data/speech_docs")

writedlm("./data/labelling/labelling_sentences.csv", Iterators.flatten(([names(labelling_sentences)], eachrow(labelling_sentences))), ',')
