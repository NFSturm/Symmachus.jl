module WordFocus

using Distributions: Beta
using MLStyle.Modules.Cond

export focus_scaler, generate_word_focus_distribution, generate_syntactic_window

@doc """
This function generates a scale adjustment to stretch the focus distribution to
prevent too tight sampling densities. It is calculated as the sentence length
divided by the index of the sentence root. Finally, we multiply this value by two.
"""
function focus_scaler(sentence_root_index::Int64, sentence_length::Int64)::Int64
    return Int(ceil(sentence_length/sentence_root_index)*2)
end

@doc """
This functions generates a beta distribution that can be used for sampling from
a random variable to emulate attention.
"""
function generate_word_focus_distribution(sentence_root_index::Int64, sentence_length::Int64)
    scale_adjustment = focus_scaler(sentence_root_index, sentence_length)::Union{Float64, Int64}
    dist = 1 + sentence_length*Beta(sentence_root_index + scale_adjustment, sentence_length)
    return dist
end


@doc """
This function generates a syntactic window specified padding.
"""
function generate_syntactic_window(token_index::Int64, doc_length::Int64, window_pad::Int64)::Vector{Int64}
    res = @cond begin
        token_index - window_pad > 1 && token_index + window_pad < doc_length => token_index - window_pad:1:token_index + window_pad
        token_index - window_pad < 1 && token_index + window_pad < doc_length => 1:1:token_index + window_pad
        token_index - window_pad > 1 && token_index + window_pad > doc_length => token_index - window_pad:1:doc_length
    end
    return res
end

end
