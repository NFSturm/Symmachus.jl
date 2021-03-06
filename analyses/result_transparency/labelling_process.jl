using CSV
using StatsBase
using DataFrames
using DataFramesMeta
using Serialization

#******* DESERIALIZING LABELLING PROCESS RESULTS ********
labelling_process_results = deserialize("./data/labels/labelled_data_final_21102021.jls")

#[291966, 282405. 459641, 147016, 447045, 462333, 783968, 627807, 2560, 274177]
row_samples = sample(1:nrow(labelling_process_results), 10)
serialize(joinpath(@__DIR__, "row_samples.jls"), row_samples)

sample_subset = labelling_process_results[row_samples, :]
