module Lemmatization

using Chain

export lemmatize

@doc """
    lemmatize(text::String)::String

Lemmatizes Portuguese plural nouns, based on rules defined in the body.
"""
function lemmatize(text::String)::String
    @chain text begin
        replace(_, r"([ao])s" => s"\1")
        replace(_, r"([eao])is" => s"\1l")
        replace(_, r"[ãõ][eo]s" => s"ão")
        replace(_, r"([rzns])es\s*" => s"\1")
    end
end

end
