module kNN

    using NearestNeighbors
    using Distances

    export compute_nearest_neighbours

    @doc """
        compute_nearest_neighbours(comparison_point::Vector{Float32}, data::Vector{Vector{Float32}}, num_neighbors::Int64)::Tuple{Int64, Float32}

    Computes the `num_neighbors` from `data`. The data is supposed to be in the \n
    `encoding_column`. Returns the indices and distances.
    """
    function compute_nearest_neighbours(comparison_point::Vector{Float32}, data::Vector{Vector{Float32}}, num_neighbors::Int64)::Tuple{Vector{Int64}, Vector{Float32}}

        # Creating data matrices
        data_matrix = hcat(data...)

        # Instantiating the splitting tree
        balltree = BallTree(data_matrix, Euclidean(); reorder = false)

        # Computing indices and distances
        # Catching the case when there are more requested neighbors than data points
        idxs, dists = knn(balltree, comparison_point, min(num_neighbors, size(data_matrix)[2]), true)

        sims = 1 .- cosine_dist.(Ref(comparison_point), data[idxs])

        idxs, sims
    end

end
