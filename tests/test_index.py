"""Checks the saved FAISS index without calling the OpenAI API."""

import numpy as np

ADA_002_DIM = 1536


def test_index_size_matches_corpus(index, records):
    assert index.ntotal == len(records)
    assert index.d == ADA_002_DIM


def test_each_vector_retrieves_itself_first(index):
    # A stored embedding should be its own nearest neighbor at distance ~0.
    for i in range(index.ntotal):
        vec = index.reconstruct(i).reshape(1, -1)
        distances, ids = index.search(vec, 1)
        assert ids[0][0] == i
        assert distances[0][0] < 1e-4


def test_top_k_is_sorted_and_unique(index):
    query = index.reconstruct(0).reshape(1, -1) + np.float32(0.01)
    distances, ids = index.search(query, 5)
    assert list(distances[0]) == sorted(distances[0])
    assert len(set(ids[0])) == 5
