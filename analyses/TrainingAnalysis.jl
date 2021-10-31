using DataFrames
using DataFramesMeta
using StatsBase
using ColorSchemes
using Compose
using Fontconfig
using Gadfly
using Cairo
using Serialization
using Pipe
using Revise

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

performance_plot = plot(performance_dataframe,
    x=:epoch,
    y=:f1_score,
    Geom.point,
    Geom.line,
    Guide.xlabel("Epoch No."),
    Guide.ylabel("F1 Score on Test Set"),
    Theme(
        default_color=colorant"#4169e1",
        line_width=1mm,
        point_size=2mm,
        minor_label_font_size=12pt,
        major_label_font_size=15pt
    )
)
performance_plot = plot(performance_dataframe,
    x=:epoch,
    y=:f1_score,
    Geom.line,
    Guide.xlabel("Epoch No."),
    Guide.ylabel("F1 Score on Test Set"),
    Theme(
        default_color=colorant"#4169e1",
        line_width=1.25mm,
        minor_label_font_size=12pt,
        major_label_font_size=15pt
    )
)

draw(PNG("./plots/labelling_performance.png", 10inch, 6inch), performance_plot)

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

    stack(args_dataframe,
        [:max_discourse_context_size, :max_sentence_context_size, :self_weight],
        variable_name=:param_name,
        value_name=:param_value
    )
end

symmachus_args_dataframe = create_parameter_dataframe(model_history)

param_subset = @rsubset symmachus_args_dataframe begin
    :param_name in ["max_discourse_context_size", "max_sentence_context_size"]
end

context_plot = plot(
    param_subset,
    x=:epoch,
    y=:param_value,
    color=:param_name,
    Geom.line,
    Guide.xlabel("Epoch No."),
    Guide.ylabel("Parameter Value"),
    Guide.colorkey(title="Parameter Names", labels=["Max Discourse Context", "Max Sentence Context"]),
    Scale.color_discrete(n -> get(ColorSchemes.tol_muted, range(0,1, length=n))),
    Theme(
        line_width=1.25mm,
        minor_label_font_size=12pt,
        major_label_font_size=15pt,
        key_title_font_size=14pt,
        key_label_font_size=12pt
    )
)

draw(PNG("./plots/symmachus_args.png", 10inch, 6inch), context_plot)


param_subset_self_weight = @rsubset symmachus_args_dataframe begin
    :param_name .== "self_weight"
end

self_weight_plot = plot(
    param_subset_self_weight,
    x=:epoch,
    y=:param_value,
    color=:param_name,
    Geom.line,
    Guide.xlabel("Epoch No."),
    Guide.ylabel("Parameter Value"),
    Guide.colorkey(title="Parameter Names", labels=["Self Weight"]),
    Scale.color_discrete(n -> get(ColorSchemes.Set2_3, range(1, length=n))),
    Theme(
        line_width=1.25mm,
        minor_label_font_size=12pt,
        major_label_font_size=15pt,
        key_title_font_size=14pt,
        key_label_font_size=12pt
    )
)

draw(PNG("./plots/symmachus_args_2.png", 10inch, 6inch), self_weight_plot)
