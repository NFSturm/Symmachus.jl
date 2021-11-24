using DataFrames
using DataFramesMeta
using StatsBase
using Compose
using Fontconfig
using CairoMakie
using Serialization
using Pipe
using Revise

CairoMakie.activate!()

#********************** DEFINING STRUCTS FOR DESERIALIZATION ******************

mutable struct SymmachusArgs
    max_discourse_context_size::Int64
    max_sentence_context_size::Int64
    self_weight::Number
end

mutable struct BoostingArgs
    num_rounds::Int64 # Number of rounds for training the booster
    metrics::Vector{String} # The metric to be chosen
    params::Vector{Pair{String, Any}} # Model parameters
    true_threshold::Float64 # Threshold for positive prediction
    train_prop::Float64 # Proportion of observations to be used for training
end

#********************** DESERIALIZING MODEL HISTORY ****************************

model_history = deserialize("./cache/final_model/model_history.jls")

#********************** VISUALIZING BOOSTER PERFORMANCE ************************


@doc """
    create_performance_dataframe(model_history::Vector{Dict})::DataFrame

Creates a DataFrame with model performance from `model_history`.
"""
function create_performance_dataframe(model_history::Vector{Dict})::DataFrame

    performance_dataframe = DataFrame(
                    epoch = Int64[],
                    f1_score = Float64[]
                    )

    performance_history = [(i, epoch[:performance]) for (i,epoch) in enumerate(model_history)]

    for row in performance_history
        push!(performance_dataframe, row)
    end

    performance_dataframe
end

performance_dataframe = create_performance_dataframe(model_history)

using CairoMakie
CairoMakie.activate!()

labelling_theme() = Theme(
    fontsize=20, font="Crimson",
    Axis=(xlabelsize=20, xgridstyle=:dash, ygridstyle=:dash,
        xtickalign=1, ylabelsize=20, ytickalign=1, yticksize=10, xticksize=10,
        xlabelpadding=-5, xlabel="Epoch No.", ylabel="F1 Score on Test Set"),
    Legend=(framecolor=(:black, 0.5), bgcolor=(:white, 0.5))
)

performance_plot() = scatterlines(
    performance_dataframe[:, :epoch],
    performance_dataframe[:, :f1_score];
    color="#155ff0",
    linewidth=2,
    markercolor="#155ff0")

plot1 = with_theme(labelling_theme()) do
    performance_plot()
end

save("./plots/labelling_performance.pdf", plot1)

#********************** VISUALIZING SELF-TRAINING PARAMS ***********************

@doc """
    get_parameter_history(model_history::Vector{Dict}, struct_name::Symbol, parameter_name::Symbol)

Returns the parameter history of a `parameter_name` from `model_history`
"""
function get_parameter_history(model_history::Vector{Dict}, struct_name::Symbol, parameter_name::Symbol)
    struct_array = [epoch[struct_name] for epoch in model_history]
    parameter_history = [getfield(arg, parameter_name) for arg in struct_array]
    parameter_history
end


@doc """
    create_parameter_dataframe(model_history::Vector{Dict})::DataFrame

Creates a DataFrame with the parameters of the model from `model_history`.
"""
function create_parameter_dataframe(model_history::Vector{Dict})::DataFrame

    args_dataframe = DataFrame(
            epoch = Int[],
            max_discourse_context_size = Int64[],
            max_sentence_context_size = Int64[],
            self_weight = Float64[]
    )

    epoch_array = 1:30 |> collect

    max_discourse_context_size_array = get_parameter_history(model_history, :symmachus_args, :max_discourse_context_size)

    max_sentence_context_size_array = get_parameter_history(model_history, :symmachus_args, :max_sentence_context_size)

    self_weight_array = get_parameter_history(model_history, :symmachus_args, :self_weight)

    rows = zip(epoch_array, max_discourse_context_size_array, max_sentence_context_size_array, self_weight_array)

    for row in rows
        push!(args_dataframe, row)
    end

    #stack(args_dataframe,
    #    [:max_discourse_context_size, :max_sentence_context_size, :self_weight],
    #    variable_name=:param_name,
    #    value_name=:param_value
    #)

    args_dataframe
end

symmachus_args_dataframe = create_parameter_dataframe(model_history)

context_theme() = Theme(
    fontsize=20, font="Crimson",
    Axis=(xlabelsize=20, xgridstyle=:dash, ygridstyle=:dash,
        xtickalign=1, ylabelsize=20, ytickalign=1, ylabelpadding=2, yticksize=10, xticksize=10,
        xlabelpadding=-5, xlabel="Epoch No.", ylabel="Parameter Value"),
    Legend=(framecolor=(:black, 0.5), bgcolor=(:white, 0.5))
)

function plot_context()
    fig, ax, _ = scatterlines(
        symmachus_args_dataframe[:, :epoch],
        symmachus_args_dataframe[:, :max_discourse_context_size];
        label="Max. Discourse Context Size",
        color="#343978",
        linewidth=2,
        markercolor="#343978"
    )
    scatterlines!(
        symmachus_args_dataframe[:, :epoch],
        symmachus_args_dataframe[:, :max_sentence_context_size];
        label="Max. Sentence Context Size",
        color="#a7b9d6",
        linewidth=2,
        markercolor="#a7b9d6"
    )
    axislegend("Parameter Values", position=:rb)
    fig
end

context_plot = with_theme(context_theme()) do
    plot_context()
end

save("./plots/symmachus_args.pdf", context_plot)

function plot_self_weight()
    fig, ax, _ = scatterlines(
        symmachus_args_dataframe[:, :epoch],
        symmachus_args_dataframe[:, :self_weight];
        label="Self-Weight",
        color="#2f4e68",
        linewidth=2.5,
        markercolor="#2f4e68"
    )
    axislegend("Parameter Values", position=:rb)
    fig
end

self_weight_plot = with_theme(context_theme()) do
    plot_self_weight()
end

save("./plots/self_weight.pdf", self_weight_plot)
