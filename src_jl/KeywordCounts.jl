using CSV
using DataFrames
using Chain
using Parquet
using ThreadsX
using Unicode: normalize

@doc """
    normalize_string(string::String) = normalize(string, stripmark=true)

Simple wrapper around `normalize` to normalize a string.
"""
normalize_string(string::String) = normalize(string, stripmark=true)

#**** Loading the required DataFrames ****

keywords = CSV.File("/home/nfsturm/Dev/Symmachus.jl/data/keywords/keyword_list.csv") |> DataFrame

speeches = DataFrame(read_parquet("/home/nfsturm/Dev/Symmachus.jl/data/speeches/ddr_dataframe_20210829.parquet"))

#deputies = DataFrame(read_parquet("/Users/nfsturm/policy-monitoring-backend/data/deputies/deputies_20210816.parquet"))

# Selecting only the keywords column
sdg_keywords = keywords[!, :verbetes]

@doc """
    process_keywords(keywords_string::String)::Vector{String}

Processes `keywords` and returns a vector of strings.
"""
function process_keywords(keywords_string::String)::Vector{String}
    keywords_clean = @chain keywords_string begin
        replace(_, "\'"=>"")
        replace(_, "_" => " ")
    end
    String.(split(keywords_clean, ","))
end

sdg_keywords_clean = process_keywords.(sdg_keywords)

function count_keywords(keywords::Vector{String}, text::String)::Int64
    sum(occursin.(keywords, text))
end

@doc """
    make_sdg_count_column(keywords::Vector{String}, speeches::DataFrame)::Vector{Int64}

Counts the occurrence of selected keywords in a DataFrames row.
"""
function make_sdg_count_column(keywords::Vector{String}, speeches::DataFrame)::Vector{Int64}
    ThreadsX.map(eachrow(speeches)) do row
        count_keywords(keywords, row[:text])
    end
end

@doc """
    construct_complete_sdg_count_columns(keywords::Vector{Vector{String}}, sppeches::DataFrame)::Vector{Vector{Int64}}

Constructs a DataFrames column with keyword counts for a particular list of `keywords` as the row value.
"""
function construct_complete_sdg_count_columns(keywords::Vector{Vector{String}}, speeches::DataFrame)::Vector{Vector{Int64}}
    counts = Vector{Int64}[]
    for keyword_vector in keywords
        sdg_count = make_sdg_count_column(keyword_vector, speeches)
        push!(counts, sdg_count)
    end
    return counts
end

keyword_counts = construct_complete_sdg_count_columns(sdg_keywords_clean, speeches)

new_columns = collect(zip(1:length(sdg_keywords), keyword_counts))

function add_columns(column_data::Vector{Tuple{Int64, Vector{Int64}}}, dataframe::DataFrame)::DataFrame
    for column in column_data
        dataframe[!, "sdg_$(column[1])"] = column[2]
    end
    return dataframe
end

speech_dataframe = add_columns(new_columns, speeches)

CSV.write("./data/keyword_counts/sdg_keyword_count.csv", speech_dataframe)

#=
speech_dataframe_normalized = transform(speech_dataframe, [:name] .=> ByRow(x -> normalize_string(x)) .=> [:normalized_name])

deputies = transform(deputies, [:name] .=> ByRow(x -> normalize_string(x)) .=> [:normalized_name])
deputies_clean = select(deputies, Not(:name))

speech_dataframe_w_deputies = innerjoin(speech_dataframe_normalized, deputies_clean, on=:normalized_name)

CSV.write("sdg_keyword_count.csv", speech_dataframe_w_deputies)
=#
