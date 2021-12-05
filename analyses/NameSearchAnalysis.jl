using CSV
using Pipe
using Parquet
using CairoMakie
using DataFrames
using DataFramesMeta
using Serialization
using Transducers
using Statistics
using Revise

include("../src_jl/EncodingUtils.jl")
using .EncodingUtils

include("../src_jl/DeputyMetaInfo.jl")
using .MetaInfo

name_search_results = [deserialize("./search_cache/search_cache_names$(i).jls") for i in 1:6]

filtered_search_results = name_search_results |> Iterators.flatten |> Filter(!isnothing) |> collect

deputy_meta_info = DataFrame(CSV.File("./analyses/deputies.csv"))

relevance_info = getindex.(filtered_search_results, 2)

result_container = append_deputy_meta_info(deputy_meta_info, relevance_info)

deputy_df = @pipe DataFrame(result_container) |>
                @transform(_, :relevances = mean.(:relevances))

speech_act_relevance_corr = cor(deputy_df[:, :nrow_speech_acts], deputy_df[:, :relevances])

activity_relevance_corr = cor(deputy_df[:, :nrow_activities], deputy_df[:, :relevances])

#********************* PLOT LENGTHS VS RELEVANCE ********************

function generate_context_theme(axis_labels::Tuple{String, String})
    Theme(
        fontsize=20, font="Crimson",
        Axis=(xlabelsize=20, xgridstyle=:dash, ygridstyle=:dash,
            xtickalign=1, ylabelsize=20, ytickalign=1, ylabelpadding=2, yticksize=10, xticksize=10,
            xlabelpadding=-5, xlabel=axis_labels[1], ylabel=axis_labels[2]),
        Legend=(framecolor=(:black, 0.5), bgcolor=(:white, 0.5))
    )
end

theme1 = generate_context_theme(("No. Speech Acts (Log)", "Query Result Relevance (0-1)"))

speech_act_plot = with_theme(theme1) do
    scatter(
        log.(deputy_df[:, :nrow_speech_acts]),
        deputy_df[:, :relevances];
        color="#1874cd"
        )
end

save("./plots/speech_act_plot.pdf", speech_act_plot)

theme2 = generate_context_theme(("No. Parliamentary Activities (Log)", "Query Result Relevance (0-1)"))

activity_plot = with_theme(theme2) do
    scatter(
        log.(deputy_df[:, :nrow_activities]),
        deputy_df[:, :relevances];
        color="#24536C"
        )
end

save("./plots/activity_plot.pdf", activity_plot)
