import re
from pathlib import Path
import numpy as np
from typing import List
from annoy import AnnoyIndex

def sort_file_paths(path: str) -> List[str]:
    """
    Params:
    ------------------
    path: str
      The path of the directory

    Returns:
    -----------------
    List[str]
       A list of paths in string format, in ascending file number order
    
    """
    string_paths = [str(path) for path in list(Path(path).iterdir())]
    string_paths.sort(key= lambda x: int(re.findall(r"\d+", x)[0]))
    return string_paths


def retrieve_numpy_arrays(file_paths: List[str]):
    """
    Params:
    --------------
    file_paths: List[str]
      File paths pointing to numpy array files (*.npz)

    Returns:
    --------------
    A list of numpy arrays

    """

    embedding_arrays = []

    for path in file_paths:
        arr = np.load(path)
        embedding_arrays.append(arr)

    return embedding_arrays


def build_search_index(embeddings: List[np.ndarray], embedding_size: int) -> AnnoyIndex:
    """
    Params:
    -------------
    embeddings: List[np.ndarray]
      A list of vector embeddings

    embedding_size: int
      The size of of the vector embedding

    Returns:
    -------------
    AnnoyIndex:
      An index object
    """

    index = AnnoyIndex(embedding_size, 'angular') 

    for i in range(0,len(embeddings)):
        index.add_item(i+1, embeddings[i])
    
    index.build(1_250, n_jobs=8)

    return index


if __name__ == '__main__':

    filepaths_sorted = sort_file_paths("./embedding_arrays/")

    embedding_arrays = retrieve_numpy_arrays(filepaths_sorted)

    index = build_search_index(embedding_arrays, 768)

    index.save("./index/index.ann")



