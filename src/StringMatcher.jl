module StringMatcher

export get_nearest_word_substitute, substitute_word_with_nearest_neighbour

using StringDistances
using MLStyle.Modules.Cond

function get_nearest_word_substitute(word::String, word_lookup::Dict{String, Int64})::String
    return findnearest(word, collect(keys(word_lookup)), RatcliffObershelp())[1]
end

function substitute_word_with_nearest_neighbour(word::String, word_lookup_table::Dict{String, Int64})
    substitute_word = @cond begin
        word ∈ keys(word_lookup_table) => word
        word ∉ keys(word_lookup_table) => get_nearest_word_substitute(word, word_lookup_table)
    end
    return substitute_word
end


end
