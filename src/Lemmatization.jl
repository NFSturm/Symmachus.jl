module Lemmatization

using Chain
using Pipe
using MLStyle.Modules.Cond
using Unicode

export lemmatize

@doc """
    lemmatize(text::String)::String

Lemmatizes Portuguese plural nouns, based on rules defined in the body.
"""
function lemmatize(text::String)::String
    @chain text begin
        Unicode.normalize(_, stripmark=true)
        replace(r"([aoAO])s\s{1,}" => s"\1 ")
        replace(r"([eao])is" => s"\1l")
        replace(r"[ãõ][eo]s" => s"ão")
        replace(r"([rzns])es\s{1,}" => s"\1")
    end
end

end
