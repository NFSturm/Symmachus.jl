using CSV
using DataFrames
using DataFramesMeta
using Statistics
using Serialization
using Pipe
using Chain
using CairoMakie
using ColorSchemes
using Transducers

topics_over_time_results = deserialize("./search_cache/all_year_topic_search.jls")

@doc """
    summarize_yearly_alignment(search_results::Vector{DataFrame})::Vector{DataFrame}

Summarizes yearly results for every year in `search_results`.
"""
function summarize_yearly_alignment(search_results::Vector{DataFrame})::Vector{DataFrame}
    map(topics_over_time_results) do df
        @chain df begin
            groupby(_, :party)
            @combine _ begin
                :sdgs_year_summary = mean(:mean_alignment_per_party)
                :year
            end
            unique(_, :party)
        end
    end
end

yearly_series = summarize_yearly_alignment(topics_over_time_results)

global_alignment = @chain summarize_yearly_alignment(topics_over_time_results) begin
    vcat(_...)
    groupby(_, :party)
    @combine _ begin
        :mean_sdgs_alignment = mean(:sdgs_year_summary)
    end
end

function generate_timeseries_theme(axis_labels::Tuple{String, String})
    Theme(
        fontsize=25, font="Crimson",
        Axis=(xlabelsize=25, xgridstyle=:dash, ygridstyle=:dash, yminorticksvisible = true,
            xtickalign=1, ylabelsize=25, ytickalign=1, ylabelpadding=5, yticksize=10, xticksize=10,
            xlabelpadding=2, xlabel=axis_labels[1], ylabel=axis_labels[2]),
        Legend=(framecolor=(:black, 0.5), bgcolor=(:white, 0.5))
    )
end

timeseries_theme = generate_timeseries_theme(("Year", "Alignment Score"))

reduced_yearly_series = vcat(yearly_series...)

parties = @pipe reduced_yearly_series[:, :party] |>
            Set |>
            collect |>
            Filter(x -> x ∉ ["chega", "il", "independente"]) |>
            collect

function make_coordinates(yearly_series::DataFrame, party::String)
    party_subset = filter(row -> row.party == party, yearly_series)
    party_scores = @select(party_subset, :sdgs_year_summary) |> Matrix |> vec
    years = @select(party_subset, :year) |> Matrix |> vec .|> Int
    Point2f.(years, party_scores)
end

plot_data = [make_coordinates(reduced_yearly_series, party) for party in parties]

party_color_scheme = ["#93c47d", "#910c0c", "#3d85c6", "#037971"]

p1 = with_theme(timeseries_theme) do
    fig = Figure(resolution=(1000, 600))
    ax = fig[1,1] = Axis(fig)
    ax.xticks = 2009:2021
    series!(plot_data[1:4]; labels=uppercase.(parties[1:4]), linewidth=5, markersize=8, color=party_color_scheme)
    vlines!(ax, [2009, 2011, 2015, 2019]; linewidth=2, color="#989c9c")
    axislegend("Party Name", position=:rb)
    fig
end

save("./plots/party_alignment1.pdf", p1)

party_color_scheme2 = ["#e92d56", "#910c0c", "#ff7d02"]

p2 = with_theme(timeseries_theme) do
    fig = Figure(resolution=(1000, 600))
    ax = fig[1,1] = Axis(fig)
    ax.xticks = 2009:2021
    series!(plot_data[5:7]; labels=uppercase.(parties[5:7]), linewidth=5, markersize=8, color=party_color_scheme2)
    vlines!(ax, [2009, 2011, 2015, 2019]; linewidth=2, color="#989c9c")
    axislegend("Party Name", position=:rb)
    fig
end

save("./plots/party_alignment2.pdf", p2)
