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

filter_judgements(data::DataFrame) = @subset(data, :label .== 1)

is_minister_name(name::String) = occursin("Ministr", name)

is_secretary_name(name::String) = occursin("cretário", name)

deputy_activity = @pipe read_deputy_activity("./data/deputy_activity") |>
    transform(_, :time .=> ByRow(x -> format_unix_date_column(x, 1:10)) .=> :datetime)


labelled_data = @pipe read_labelled_data("./data/broadcast.csv") |>
    transform(_, :sentence_text => ByRow(x -> replace(x, r"\d{1,2}\s*DE\s*[A-Z]{4,9}\s*DE\s*\d{4}" => s"")) => :sentence_text)

labelled_data_filtered = filter_judgements(labelled_data)

CSV.write("./data/datasets/labelled_data_filtered.csv", labelled_data_filtered)
CSV.write("./data/datasets/deputy_activity.csv", deputy_activity)
