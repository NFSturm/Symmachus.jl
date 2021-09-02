using JSON3
using ThreadsX
using Chain
using Serialization: serialize

include("SymmachusCore.jl")

using .SymmachusCore

# Functions for working with incoming JSONs
function parse_json(filepath::String)
    json_string = read(filepath)
    discourse_documents = JSON3.read(json_string)
    return discourse_documents
end


"""
This functions turns dictionary keys into integers
"""
function keys_to_int(dictionary)::Dict{Int64, String}
    keys = [pair[1] for pair in dictionary]::Vector{Int64}
    values = [pair[2] for pair in dictionary]::Vector{String}
    Dict(keys .=> values)
end


"""
A simple helver function to convert a specific structure to Dict
"""
function convert_to_dict(sentence)::Dict
    return convert(Dict, sentence)
end


"""
This function traverses a JSON array and converts the elements to Dict
"""
function document_traversal(discourse_documents)::Array{Dict}
    return [convert_to_dict(sentence) for sentence in discourse_documents]
end


function make_sentence(sentence_dict::Dict{Symbol, Any})::Sentence
    return Sentence(
        keys_to_int(sentence_dict[:lookup]),
        sentence_dict[:dependency_graph],
        sentence_dict[:root],
        sentence_dict[:doc_length],
        sentence_dict[:sentence_literal],
        sentence_dict[:unique_doc_identifier],
        sentence_dict[:sentence_id],
        sentence_dict[:actor_name],
        sentence_dict[:discourse_time]

    )
end


make_sentence_array(sentence_dicts::Array{Dict}) = [make_sentence(sentence) for sentence in sentence_dicts]::Vector{Sentence}

function transform_json(file_path::String)::Document
    sentences = @chain file_path begin
        parse_json
        document_traversal
        make_sentence_array
    end
    return Document(sentences)
end

"""
This functions extracts the document UUID from a sentence. The document
UUID is uniform across all sentences of a given document.
"""
function get_doc_uuid(doc::Document)::String
    return doc.sentences[1].doc_uuid
end


ThreadsX.foreach(readdir("./data/json_inputs", join=true)) do f
    document = transform_json(f)::Document
    uuid = get_doc_uuid(document)::String
    serialize("./data/speech_docs/$(uuid).jls", document)
end
