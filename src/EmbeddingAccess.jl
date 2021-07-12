module EmbeddingAccess

using Serialization: deserialize

export get_embedding_from_string, get_word_lookup_table, load_fasttext_embeddings

@doc """
Creates a lookup table for embeddings.
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
Loads FastText embeddings from disk.
"""
function load_fasttext_embeddings(data_dir::String)
    embeddings_vocab = deserialize("$(data_dir)/fasttext_pt_embeddings_vocab.jls")
    embeddings = deserialize("$(data_dir)/fasttext_pt_embeddings.jls")
    return embeddings_vocab, embeddings
end

end
