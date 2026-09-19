import pytest

from metrics import mean, precision_at_k, recall_at_k


def test_precision_counts_only_top_k():
    predicted = ["1", "2", "3", "4", "5", "6"]
    assert precision_at_k(predicted, {"1", "6"}, k=5) == pytest.approx(0.2)


def test_precision_penalizes_short_lists():
    assert precision_at_k(["1"], {"1"}, k=5) == pytest.approx(0.2)


def test_precision_rejects_bad_k():
    with pytest.raises(ValueError):
        precision_at_k(["1"], {"1"}, k=0)


def test_recall():
    assert recall_at_k(["1", "2", "3"], {"1", "9"}, k=3) == pytest.approx(0.5)


def test_mean():
    assert mean([0.2, 0.4]) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        mean([])
