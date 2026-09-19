"""Retrieval metrics used by the evaluation scripts.

Kept free of API calls so they can be unit tested without an OpenAI key.
"""


def precision_at_k(predicted, relevant, k=5):
    """Share of the top-k predictions that are relevant.

    ``predicted`` is a ranked sequence of PMIDs. Only the first ``k`` count,
    and the denominator is always ``k`` so a short result list is penalized.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = list(predicted)[:k]
    hits = sum(1 for pmid in top_k if pmid in relevant)
    return hits / k


def recall_at_k(predicted, relevant, k=5):
    """Share of the relevant PMIDs that appear in the top-k predictions."""
    if not relevant:
        raise ValueError("relevant set is empty")
    top_k = set(list(predicted)[:k])
    return len(top_k & set(relevant)) / len(relevant)


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("no values to average")
    return sum(values) / len(values)
