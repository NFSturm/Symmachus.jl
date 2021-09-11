module LabelingUtils

using CSV
using DataFrames

export read_label_data

function read_label_data(path::String)
    DataFrame(CSV.File(path))
end

end
