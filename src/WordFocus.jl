module WordFocus

using Distributions: Beta
using MLStyle.Modules.Cond

export focus_scaler, generate_word_focus_distribution, generate_syntactic_window

@doc """
    focus_scaler(sentence_root_index::Int64, sentence_length::Int64)::Int64

This function generates a scale adjustment to stretch the focus distribution to
prevent too tight sampling densities. It is calculated as the `sentence_length`
divided by the index of the `sentence_root`. Finally, we multiply this value by two.
"""
function focus_scaler(sentence_root_index::Int64, sentence_length::Int64)::Int64
    return Int(ceil(sentence_length/sentence_root_index)*2)
end

@doc """
    generate_word_focus_distribution(sentence_root_index::Int64, sentence_length::Int64)

This is the base mechanism to emulate **attention** in a lightweight way. The main ingredient is a scaled \n
and shifted Beta distribution, from which a single sample is taken. Said distribution is scaled in such a \n
way that words closer to the sentence root a sampled more frequently, thus exposing the assumption that \n
the sentence is root is that part of a sentence that captures the reader's attention more easily to di- \n
gest the information contained in the sentence.

This mechanism needs the `sentence_root_index` as well as the overall `sentence_length`.
"""
function generate_word_focus_distribution(sentence_root_index::Int64, sentence_length::Int64)
    scale_adjustment = focus_scaler(sentence_root_index, sentence_length)::Union{Float64, Int64}
    dist = 1 + sentence_length*Beta(sentence_root_index + scale_adjustment, sentence_length)
    return dist
end


@doc """
    generate_syntactic_window(token_index::Int64, doc_length::Int64, window_pad::Int64)::Vector{Int64}

Given a `token_index` and `sentence_length` a syntactic window, comprising the indices near the `token_index`, is generated. \n
The width of the syntanctic window can be changed by setting `window_pad`.
"""
function generate_syntactic_window(token_index::Int64, sentence_length::Int64, window_pad::Int64)::Vector{Int64}
    res = @cond begin
        token_index - window_pad > 1 && token_index + window_pad < sentence_length => token_index - window_pad:1:token_index + window_pad
        token_index - window_pad < 1 && token_index + window_pad < sentence_length => 1:1:token_index + window_pad
        token_index - window_pad > 1 && token_index + window_pad > sentence_length => token_index - window_pad:1:sentence_length
    end
    return res
end

end
