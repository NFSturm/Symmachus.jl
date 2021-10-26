module MLUtils

using StatsBase
using UnPack
using MLStyle.Modules.Cond
using DataFrames
using Random

export confusion_matrix, precision, recall, f1_score, make_train_test_data

round_prediction(pred::Float64, threshold::Float64) = pred > threshold ? 1 : 0


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
	pred_labels = round_prediction.(predictions, Ref(true_threshold))
	confusion_base = eval_obs.(pred_labels, true_labels) |> countmap

	# Unpacking the confusion matrix elements
	@unpack TP, FP, TN, FN = confusion_base

	confusion_array = [TP, FP, FN, TN]

	reshape(confusion_array, (2,2))

end


@doc """
Calculates the precision of a model based on the confusion matrix.
"""
function precision(confmat::Matrix{Int64})
	confmat[1, 1] / ( confmat[1, 1] + confmat[2, 1] )
end


@doc """
Calculates the recall of a model based on the confusion matrix.
"""
function recall(confmat::Matrix{Int64})
	confmat[1, 1] / ( confmat[1, 1] + confmat[1, 2] )
end


@doc """
Calculates the F1 score based on the confusion matrix.
"""
function f1_score(confmat::Matrix{Int64})
	2 * ( precision(confmat) * recall(confmat) ) / ( precision(confmat) + recall(confmat) )
end


@doc """
    make_train_test_data(all_data::DataFrame, label_column::String, feature_column::String, train_prop::Float64)

Randomly shuffles `all_data` according to `train_prop`. Returns the training and test data including the corresponding labels.
"""
function make_train_test_data(all_data::DataFrame, feature_column::String, label_column::String, train_prop::Float64)

	all_labels = all_data[!, label_column]
	all_features = all_data[!, feature_column]

	data_length = nrow(all_data)

	indices = shuffle(1:1:data_length)

	split_index = Int(floor(data_length * train_prop))

	train_indices = indices[1:split_index]
	test_indices = indices[split_index+1:end]

	X_train, y_train = convert(Array{Float64}, transpose(hcat(all_features[train_indices]...))), convert(Array{Int64}, all_labels[train_indices])
	X_test, y_test = convert(Array{Float64}, transpose(hcat(all_features[test_indices]...))), convert(Array{Int64}, all_labels[test_indices])

	return X_train, y_train, X_test, y_test

end

end
