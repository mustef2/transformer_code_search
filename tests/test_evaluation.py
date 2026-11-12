
from code_search.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)


def test_precision_at_k():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "E"}

    assert precision_at_k(retrieved, relevant, 3) == 2 / 3


def test_recall_at_k():
    retrieved = ["A", "B", "C"]
    relevant = {"A", "C", "D"}

    assert recall_at_k(retrieved, relevant, 3) == 2 / 3


def test_reciprocal_rank():
    retrieved = ["A", "B", "C"]
    relevant = {"C"}

    assert reciprocal_rank(retrieved, relevant) == 1 / 3


def test_reciprocal_rank_no_relevant_result():
    retrieved = ["A", "B", "C"]
    relevant = {"D"}

    assert reciprocal_rank(retrieved, relevant) == 0.0


def test_mean_reciprocal_rank():
    results = [
        (["A", "B", "C"], {"A"}),
        (["A", "B", "C"], {"B"}),
    ]

    assert mean_reciprocal_rank(results) == 0.75
