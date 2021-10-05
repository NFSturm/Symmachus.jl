using DataFrames
using CSV
using Dates
using Parquet
using Pipe

@doc """
    read_deputy_activity(data_dir::String)::DataFrame

Reads the deputy activity data from `data_dir`. Returns a DataFrame.
"""
function read_deputy_activity(data_dir::String)::DataFrame
    activity_file = readdir(data_dir, join=true) |> first
    activity = DataFrame(read_parquet(activity_file))
end

deputy_activity = read_deputy_activity("./data/deputy_activity")

@doc """
This function is necessary when the date column is in faulty UNIX time.
"""
function format_unix_date_column(unix_date::Int64, index_range::UnitRange)::Date
    @pipe unix_date |> string |> _[index_range] |> parse(Int, _) |> unix2datetime
end

transform!(deputy_activity, :time .=> ByRow(x -> format_unix_date_column(x, 1:10)) .=> :datetime)

@doc """
    read_labelled_data(data_dir::String)::DataFrame

Reads labelled data from `data_dir`. Returns a DataFrame.
"""
function read_labelled_data(data_dir::String)::DataFrame
    label_file = readdir(data_dir, join=true) |> first
    labelled_data = DataFrame(CSV.File(label_file))
end

@doc """
    parse_date_column(data::DataFrame, date_column::String)::DataFrame

Parses `date_column` in `data`. Data is expected to follow "y-m-d" convention.
"""
@everywhere function parse_date_column(data::DataFrame, date_column::String)::DataFrame
    transform(data, Symbol(date_column) => ByRow(x -> Date(x, "y-m-d")) => Symbol(date_column))
end

labelled_data = @pipe read_labelled_data("./data/testing/labelled_data.csv") |> parse_date_column(_, :discourse_time)
