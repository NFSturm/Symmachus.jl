using Pkg

Pkg.activate(".")

using CSV
using DataFrames
using Serialization
using DataFramesMeta
using Dates
using Pipe
using Revise

include("TopicSearchFunctions.jl")
using .TopicSearch

include("../src_jl/DeputyMetaInfo.jl")
using .MetaInfo

include("EncodingUtils.jl")
using .EncodingUtils

@doc """
    retrieve_topic_vectors(path::String)

Retrieves topic vectors from a `path` and unpacks numpy arrays.
"""
function retrieve_topic_vectors(path::String)
    topic_vector_paths = readdir(joinpath(path), join=true)

    orderings = @pipe split.(topic_vector_paths, "sdg") .|> split(last(_), ".") .|> first .|> parse(Int, _)

    ordered_paths = @pipe zip(topic_vector_paths, orderings) |> collect |> sort(_, by= x -> x[2]) |> getindex.(_, 1)

    topic_vectors = unpack_numpy_array.(ordered_paths)

    topic_vectors
end

topic_vectors = retrieve_topic_vectors("./src_py/pt_arrays")

deputy_meta_info = @pipe DataFrame(CSV.File("./analyses/deputies.csv")) |>
        mapcols!(cols -> string.(cols), _)

encoded_activities = DataFrame(CSV.File("./data/encoded_datasets/activities_encoded.csv"))
transform!(encoded_activities, :encoded_activities_pt => ByRow(x -> parse_encoding(x)) => :encoded_activities_pt)
transform!(encoded_activities, :name => ByRow(x -> lowercase(x)) => :name)

encoded_speech_acts = DataFrame(CSV.File("./data/encoded_datasets/speech_acts_encoded.csv"))
transform!(encoded_speech_acts, :encoded_speech_acts_pt => ByRow(x -> parse_encoding(x)) => :encoded_speech_acts_pt)
transform!(encoded_speech_acts, :actor_name => ByRow(x -> lowercase(x)) => :actor_name)

@doc """
    topic_search_by_time(year_subset, topic_vectors::Vector{Vector{Float32}}, encoded_activities::DataFrame, encoded_speech_acts::DataFrame, deputy_meta_info::DataFrame)

Computes party topic alignment statistics for a given `year` for all topic vectors in `topic_vectors`.
"""
function topic_search_by_time(year_subset, topic_vectors::Vector{Vector{Float32}}, encoded_activities::DataFrame, encoded_speech_acts::DataFrame, deputy_meta_info::DataFrame)

    #****** SUBSETTING BY YEAR *******
    activity_date_subset = filter(row -> year(row.datetime) == year_subset, encoded_activities)

    speech_act_date_subset = filter(row -> year(row.discourse_time) == year_subset, encoded_speech_acts)

    politician_names = @pipe validate_name_integrity(speech_act_date_subset, activity_date_subset) |>
                    Set .|>
                    String |>
                    collect

    #****** COMPUTING ALIGNMENT SCORES BY YEAR *******
    year_search_results = evaluate_all_names(politician_names, (:encoded_speech_acts_pt, :encoded_activities_pt), topic_vectors, activity_date_subset, speech_act_date_subset)

    #****** SUMMARIZING RESULTS BY PARTY *******
    party_summary = summarize_by_party(year_search_results, deputy_meta_info, 17)

    party_summary_concat = vcat(party_summary...)

    insertcols!(party_summary_concat, :year => year_subset)

    party_summary_concat
end

function topic_search_all_years(years::UnitRange{Int64}, topic_vectors::Vector{Vector{Float32}}, encoded_activities::DataFrame, encoded_speech_acts::DataFrame, deputy_meta_info::DataFrame)

    results = DataFrame[]

    for year in years
        year_topic_search = topic_search_by_time(year, topic_vectors, encoded_activities, encoded_speech_acts, deputy_meta_info)
        push!(results, year_topic_search)
    end
    results
end

all_years_topic_search = topic_search_all_years(2009:2021, topic_vectors, encoded_activities, encoded_speech_acts, deputy_meta_info)

serialize("./search_cache/all_year_topic_search.jls", all_years_topic_search)
