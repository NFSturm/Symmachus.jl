module MetaInfo

    using DataFrames

    export append_deputy_meta_info

    @doc """
        append_deputy_meta_info(deputy_meta_info::DataFrame, model_info::Vector{Dict{Symbol, Any}})

    Appends information from `deputy_meta_info` to `model_info`.
    """
    function append_deputy_meta_info(deputy_meta_info::DataFrame, model_info::Vector{Dict{Symbol, Any}})
        result_container = []

        for search_result in relevance_info

            name_meta_subset = filter(row -> row.deputy_name == search_result[:name], deputy_meta_info)

            if nrow(name_meta_subset) == 0
                continue
            end

            deputy_name, party, district = first(name_meta_subset)

            deputy_info = Dict(
                :deputy_name => deputy_name,
                :party => party,
                :district => district
            )

            meta_info_merged = merge(search_result, deputy_info)
            push!(result_container, meta_info_merged)
        end

        result_container
    end

end
