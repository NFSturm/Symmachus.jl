module MLUtils

using StatsBase
using UnPack
using MLStyle.Modules.Cond

export confusion_matrix, precision, recall

round_prediction(pred::Vector{Float64}, threshold::Float64) = pred > threshold ? 1 : 0


@doc """
    eval_obs(pred_obs::Int64, true_obs::Int64)::String

Evaluates an observation in confusion matrix categories. Returns a string.
"""
function eval_obs(pred_obs::Int64, true_obs::Int64)::String
	@cond begin
		pred_obs == 1 && true_obs == 1 => "TP"
		pred_obs == 1 && true_obs == 0 => "FP"
		pred_obs == 0 && true_obs == 1 => "FN"
		pred_obs == 0 && true_obs == 0 => "TN"
	end
end

@doc """
    confusion_matrix(predictions::Vector{Float64}, true_labels::Vector{Int64}, true_threshold::Float64)::Matrix{Int64}

Creates a confusion matrix. Takes as argument the raw model `predictions`, \n
the `true_labels` as well as the `true_threshold`, above which predictions are \n
classified as positive for binary classification.
"""
function confusion_matrix(predictions::Vector{Float64}, true_labels::Vector{Int64}, true_threshold::Float64)::Matrix{Int64}
	pred_labels = round_prediction.(predictions, true_threshold)
	confusion_base = eval_obs.(pred_labels, true_labels) |> countmap

	# Unpacking the confusion matrix elements
	@unpack TP, FP, TN, FN = confusion_base

	confusion_array = [TP, FP, FN, TN]

	reshape(confusion_array, (2,2))

end

function precision(confmat::Matrix{Int64})
	confmat[1, 1] / ( confmat[1, 1] + confmat[2, 1])
end

function recall(confmat::Matrix{Int64})
	confmat[1, 1] / ( confmat[1, 1] + confmat[1, 2]
end

end
