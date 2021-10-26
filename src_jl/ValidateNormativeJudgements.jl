using Pkg; Pkg.activate(".")

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

addprocs(3)

@everywhere begin
    using Pkg; Pkg.activate(".")

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
end

@everywhere include("RakeCore.jl")
@everywhere using .RakeCore

@everywhere @doc """
    read_deputy_activity(data_dir::String)::DataFrame

Reads the deputy activity data from `data_dir`. Returns a DataFrame.
"""
function read_deputy_activity(data_dir::String)::DataFrame
    activity_file = readdir(data_dir, join=true) |> first
    activity = DataFrame(read_parquet(activity_file))
end


@everywhere @doc """
This function is necessary when the date column is in faulty UNIX time.
"""
function format_unix_date_column(unix_date::Int64, index_range::UnitRange)::Date
    @pipe unix_date |> string |> _[index_range] |> parse(Int, _) |> unix2datetime
end


@everywhere @doc """
    read_labelled_data(data_file::String)::DataFrame

Reads labelled data from `data_dir`. Returns a DataFrame.
"""
function read_labelled_data(data_file::String)::DataFrame
    labelled_data = DataFrame(CSV.File(data_file))
end


@everywhere @doc """
    make_judgement_validation_endpoints(judgement_date::Date, years::Int64)::Tuple{Date, Date}

Given a `judgement_date`, the function returns a Tuple of dates that set endpoints for `years` \n
in both directions; past and future.
"""
function make_judgement_validation_endpoints(judgement_date::Date, years::Int64)::Tuple{Date, Date}
    judgement_date - Year(years), judgement_date + Year(years)
end

@everywhere @doc """
    validate_normative_judgements(name::String, labelled_data::DataFrame, deputy_activity::DataFrame, stopwords::Vector{String})::DataFrame

Validates implicit normative statements (labelled as *1*) by generating keyword tuples from each normative statement in `labelled_data` and \n
crossreferencing this with every activity in `deputy_activity`.
"""
function validate_normative_judgements(name::String, labelled_data::DataFrame, deputy_activity::DataFrame, stopwords::Vector{String})::DataFrame

    speaker_container = DataFrame(
        actor_name = String[],
        discourse_time = Date[],
        sentence_text = String[],
        label_prop = Float64[],
        keyword_matches = [],
        activity_text = String[]
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

        # Subset activity dataset

        time_activity_subset = @subset activity_subset begin
            :datetime .> make_judgement_validation_endpoints(discourse_time, 2) |> first
            :datetime .< make_judgement_validation_endpoints(discourse_time, 2) |> last
        end

        # This loops over all rows of deputy activity and generates keywords
        for activity in eachrow(time_activity_subset)

            @unpack time, text, key, name, datetime = activity

            keyword_vectors = []

            keywords2 = @pipe rake_wrapper(text, 2, stopwords) .|> _[1]
            keywords3 = @pipe rake_wrapper(text, 3, stopwords) .|> _[1]
            push!(keyword_vectors, union(keywords2, keywords3))

            keyword_matches = []

            #= For each row (i.e. each implicit pledge), the keywords for that row
            are compared with the ones generated from an individual activity.
            =#
            kw_intersection = intersect(keywords_discourse, keyword_vectors)
            push!(keyword_matches, kw_intersection)

            keyword_matches_clean = @pipe keyword_matches |> Filter(!isempty) |> collect

            if !isempty(keyword_matches_clean)
                keyword_matches_reduced = reduce(vcat, keyword_matches_clean) # Formatting matching keywords for repository
                push!(speaker_container, [actor_name, discourse_time, sentence_text, label_prob, keyword_matches_reduced, text])
            else
                push!(speaker_container, [actor_name, discourse_time, sentence_text, label_prob, keyword_matches_clean, text])
            end

        end

        @info "Validated normative judgements for $(actor_name) – $(now())"

    end
    speaker_container
end

#=
@doc """
    validate_all_judgements(names::Vector{String}, deputy_activity::DataFrame, pledges_data::DataFrame, stopwords::Vector{String})::Vector{DataFrame}

Wrapper around *validate_judgements*. Implements a progress bar for iteration.
"""
function validate_all_judgements(names::Vector{String}, deputy_activity::DataFrame, pledges_data::DataFrame, stopwords::Vector{String})::Vector{DataFrame}

    actor_container = Vector{DataFrame}[]

    num_names = length(names)
    p = Progress(num_names, 1)

    for name in names
        actor_judgements_validated = validate_normative_judgements(name, deputy_activity, pledges_data, stopwords)
        push!(actor_container)
        next!(p)
    end

    actor_container
end

=#

@everywhere begin

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

end

tst = validate_normative_judgements("Fátima Ramos", judgement_data, deputy_activity, stopwords)

tst[:, :keyword_matches] |> Filter(!isempty) |> collect

tst[:, :keyword_matches] |> Filter(!isempty) |> collect



clean_deputy_names

valid = @subset judgement_data begin
    :actor_name .== "Fátima Ramos"
end

using TextAnalysis
using Languages

text = valid[:, :sentence_text]

crps = Corpus([make_string_document(lowercase(sent)) for sent in text])
languages!(crps, Languages.Portuguese())

m = DocumentTermMatrix(crps)

k = 2            # number of topics
iterations = 1000 # number of gibbs sampling iterations

α = 0.1      # hyper parameter
β  = 0.1       # hyper parameter

ϕ, θ  = lda(m, k, iterations, α, β)



CM = CooMatrix(crps, window=5)

CM.coom |> collect

coom(CM)

CSV.write("valid.csv", valid)

valid_act = @subset deputy_activity begin
    :name .== "Fátima Ramos"
end

CSV.write("valid_act.csv", valid_act)

date1 = tst[2, :discourse_time]

@subset deputy_activity begin
    :name .== "Adão Silva"
    :datetime .> make_judgement_validation_endpoints(date1, 2) |> first
    :datetime .< make_judgement_validation_endpoints(date1, 2) |> last
end

@doc """
    validate_all_judgements(names::Vector{String}, deputy_activity::DataFrame, judgement_data::DataFrame, stopwords::Vector{String})::Vector{DataFrame}

Wrapper around *validate_judgements*. Implements a progress bar for iteration.
"""
function validate_all_judgements(names::Vector{String}, judgement_data::DataFrame, deputy_activity::DataFrame, stopwords::Vector{String})::Vector{DataFrame}

    results = pmap(
        (name, judgement_data, deputy_activity, stopwords) -> validate_normative_judgements(name, judgement_data, deputy_activity, stopwords),
        names,
        Iterators.repeated(judgement_data),
        Iterators.repeated(deputy_activity),
        Iterators.repeated(stopwords)
    )

    @info "Validated all deputy names in "

    results
end
