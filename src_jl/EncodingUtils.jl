module EncodingUtils

    using Chain

    export parse_encoding, parse_phrases

    @doc """
        parse_encoding(encoding::String)

    Given a string `encoding` of an array, returns a float array.
    """
    function parse_encoding(encoding::String)
        @chain encoding begin
            replace(_, r"[\[\]\)\(\n]" => s"") |> strip
            split(_, " ")
            filter(!isempty, _) .|> String
            parse.(Float32, _)
        end
    end


    @doc """
        parse_phrases(phrase::String)

    Given a string `phrase` with an array of phrase strings, returns a string array.
    """
    function parse_phrases(phrase::String)
        @chain phrase begin
            replace(_, r"[\]\[\n\'\)\(]" => s"") |> strip
            split(_, ", ")
            filter(!isempty, _) .|> String
        end
    end

end
