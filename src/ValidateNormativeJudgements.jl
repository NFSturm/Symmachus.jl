using DataFrames
using DataFramesMeta
using Distributed
using CSV
using Dates
using Parquet
using Pipe
using Transducers
using ThreadsX
using UnPack
using ProgressMeter
using Revise


include("RakeCore.jl")
using .RakeCore

@doc """
    read_deputy_activity(data_dir::String)::DataFrame

Reads the deputy activity data from `data_dir`. Returns a DataFrame.
"""
function read_deputy_activity(data_dir::String)::DataFrame
    activity_file = readdir(data_dir, join=true) |> first
    activity = DataFrame(read_parquet(activity_file))
end


@doc """
This function is necessary when the date column is in faulty UNIX time.
"""
function format_unix_date_column(unix_date::Int64, index_range::UnitRange)::Date
    @pipe unix_date |> string |> _[index_range] |> parse(Int, _) |> unix2datetime
end


@doc """
    read_labelled_data(data_file::String)::DataFrame

Reads labelled data from `data_dir`. Returns a DataFrame.
"""
function read_labelled_data(data_file::String)::DataFrame
    labelled_data = DataFrame(CSV.File(data_file))
end


@doc """
    make_pledge_validation_endpoints(pledge_date::Date, time_period::Int64)::Tuple{Date}

Given a `pledge_date`, the function returns a Tuple of dates that set endpoints for `time_period` \n
in both directions; past and future.
"""
function make_pledge_validation_endpoints(pledge_date::Date, time_period::Int64)::Tuple{Date, Date}
    pledge_date - Year(time_period), pledge_date + Year(time_period)
end

@doc """
    validate_normative_judgements(name::String, labelled_data::DataFrame, deputy_activity::DataFrame)::DataFrame

Validates implicit normative statements (labelled as *1*) by generating keyword tuples from each normative statement in `labelled_data` and \n
crossreferencing this with every activity in `deputy_activity`.
"""
function validate_normative_judgements(name::String, labelled_data::DataFrame, deputy_activity::DataFrame)::DataFrame

    speaker_container = DataFrame(
        actor_name = String[],
        discourse_time = Date[],
        sentence_text = String[],
        label_prop = Float64[],
        keyword_matches = []
    )

    speaker_subset = @subset labelled_data begin
        :actor_name .== name
    end

    activity_subset = @subset deputy_activity begin
        :name .== name
    end

    for row in eachrow(speaker_subset)

        @unpack actor_name, discourse_time, sentence_text, label_prob = row

        speaker_text2 = @pipe rake_wrapper(row[:sentence_text], 2, stopwords) .|> _[1]
        speaker_text3 = @pipe rake_wrapper(row[:sentence_text], 3, stopwords) .|> _[1]

        keywords_discourse = union(speaker_text2, speaker_text3)

        keyword_vectors = []

        # This loops over all rows of deputy activity and generates keywords
        foreach(activity_subset[!, :text]) do t
            keywords2 = @pipe rake_wrapper(t, 2, stopwords) .|> _[1]
            keywords3 = @pipe rake_wrapper(t, 3, stopwords) .|> _[1]
            push!(keyword_vectors, union(keywords2, keywords3))
        end

        keyword_matches = []

        #= For each row (i.e. each implicit pledge), the keywords for that row
        are compared with the ones generated from an individual activity.
        =#
        foreach(keyword_vectors) do kw
            kw_intersection = intersect(keywords_discourse, kw)
            push!(keyword_matches, kw_intersection)
        end

        keyword_matches_clean = @pipe keyword_matches |> Filter(!isempty) |> collect

        if !isempty(keyword_matches_clean)
            keyword_matches_reduced = reduce(vcat, keyword_matches_clean) # Formatting matching keywords for repository
            push!(speaker_container, [actor_name, discourse_time, sentence_text, label_prob, keyword_matches_reduced])
        else
            push!(speaker_container, [actor_name, discourse_time, sentence_text, label_prob, keyword_matches_clean])
        end


    end
    speaker_container
end


@doc """
    validate_all_judgements(names::Vector{String}, deputy_activity::DataFrame, pledges_data::DataFrame)::Vector{DataFrame}

Wrapper around *validate_judgements*. Implements a progress bar for iteration.
"""
function validate_all_judgements(names::Vector{String}, deputy_activity::DataFrame, pledges_data::DataFrame)::Vector{DataFrame}

    actor_container = Vector{DataFrame}[]

    num_names = length(names)
    p = Progress(num_names, 1)

    for name in names
        actor_judgements_validated = validate_judgements(name, deputy_activity, pledges_data)
        push!(actor_container)
        next!(p)
    end

    actor_container
end

filter_judgements(data::DataFrame) = @subset(data, :label .== 1)

is_minister_name(name::String) = occursin("Ministr", name)

is_secretary_name(name::String) = occursin("cretário", name)

stopwords = read_stopwords("./stopwords/stopwords.txt")

deputy_activity = read_deputy_activity("./data/deputy_activity")

transform!(deputy_activity, :time .=> ByRow(x -> format_unix_date_column(x, 1:10)) .=> :datetime)

labelled_data = @pipe read_labelled_data("./data/broadcast.csv")

transform!(labelled_data, :sentence_text => ByRow(x -> replace(x, r"\d{1,2}\s*DE\s*[A-Z]{4,9}\s*DE\s*\d{4}" => s"")) => :sentence_text)

judgement_data = filter_judgements(labelled_data)

deputy_names = labelled_data[!, :actor_name] |> Set |> collect .|> string

clean_deputy_names = @pipe deputy_names |> Filter(!is_minister_name) |> Filter(!is_secretary_name) |> collect

all_judgements = validate_all_judgements(clean_deputy_names, deputy_activity, judgement_data)
