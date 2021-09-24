module Lemmatization

using Chain

export lemmatize

@doc """
    lemmatize(text::String)::String

Lemmatizes Portuguese plural nouns, based on rules defined in the body.
"""
function lemmatize(text::String)::String
    @chain text begin
        replace(_, r"os" => s"o")
        replace(_, r"as" => s"a")
        replace(_, r"ais" => s"al")
        replace(_, r"eis" => s"el")
        replace(_, r"ois" => s"ol")
        replace(_, r"ãos" => s"ão")
        replace(_, r"ões" => s"ão")
        replace(_, r"ães" => s"ão")
        replace(_, r"([rzns])es\s*" => s"\1")
    end
end

end
