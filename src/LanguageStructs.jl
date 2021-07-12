module LanguageStructs

export Sentence, Document

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

end
