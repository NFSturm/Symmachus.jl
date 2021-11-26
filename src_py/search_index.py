import re
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex


def encode_query(query: str, query_keyword_separator: str, model: SentenceTransformer) -> np.ndarray:

    query_clean = " ".join(query.split(query_keyword_separator))

    encoded_query = model.encode(query_clean)

    return encoded_query


def sort_file_paths(path: str) -> List[str]:
    string_paths = [str(path) for path in list(Path("../embedding_arrays").iterdir())]
    string_paths.sort(key= lambda x: int(re.findall(r"\d+", x)[0]))
    return string_paths


def retrieve_numpy_arrays(file_paths: List[str]):

    embedding_arrays = []

    for path in file_paths:
        arr = np.fromfile(path)
        embedding_arrays.append(arr)

    return embedding_arrays


def build_search_index(embeddings: np.ndarray) -> AnnoyIndex:

    index = AnnoyIndex(len(embeddings), 'angular') 

    for emb in embeddings:
        index.add_item(index, emb)
    
    index.build(10_000, n_jobs=-1)

    return index


if __name__ == '__main__':
    ...


