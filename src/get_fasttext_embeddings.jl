using Embeddings
using Serialization

include("SymmachusCore.jl")

embeddings = load_embeddings(FastText_Text{:pt}; max_vocab_size=3_000_000)

using SymmachusCore:
    get_word_lookup_table,
    get_embedding_from_string,
    load_fasttext_embeddings

serialize("./data/embeddings/fasttext_pt_embeddings.jls", embeddings.embeddings)

serialize("./data/embeddings/fasttext_pt_embeddings_vocab.jls", embeddings.vocab)
