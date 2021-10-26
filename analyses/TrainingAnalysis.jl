using DataFrames
using StatsBase
using Gadfly
using Cairo
using Serialization
using Pipe

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

draw(PNG("./plots/labelling_performance.png", 10inch, 6inch), performance_plot)
