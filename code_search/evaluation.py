
def precision_at_k(retrieved, relevant, k=5):
    """Calculate the proportion of top-k results that are relevant."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant:
        return 0.0

    top_k = retrieved[:k]
    relevant_count = sum(1 for item in top_k if item in relevant)

    return relevant_count / k


def recall_at_k(retrieved, relevant, k=5):
    """Calculate the proportion of relevant items found in top-k."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant:
        return 0.0

    top_k = retrieved[:k]
    relevant_count = sum(1 for item in top_k if item in relevant)

    return relevant_count / len(relevant)


def reciprocal_rank(retrieved, relevant):
    """Return the reciprocal rank of the first relevant result."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(search_results):
    """Calculate mean reciprocal rank across multiple queries."""
    if not search_results:
        return 0.0

    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in search_results
    ]

    return sum(scores) / len(scores)
