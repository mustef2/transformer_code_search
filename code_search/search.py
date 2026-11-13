
import math
import re
from collections import Counter

import numpy as np


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text and split camelCase and snake_case identifiers."""
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    tokens = [t.lower() for t in TOKEN_PATTERN.findall(expanded)]

    return tokens + [
        part
        for token in tokens
        for part in token.split("_")
        if part
    ]


def bm25_scores(
    query: str,
    documents: list[dict],
    k1: float = 1.5,
    b: float = 0.75
) -> np.ndarray:
    """Calculate BM25 lexical relevance and normalize to [0, 1]."""

    if not documents:
        return np.array([], dtype=float)

    corpus = [
        tokenize(
            f"{d['qualified_name']} {d['signature']} "
            f"{d['docstring']} {d['source']}"
        )
        for d in documents
    ]

    query_terms = set(tokenize(query))
    lengths = [len(tokens) for tokens in corpus]
    avg_len = sum(lengths) / max(len(lengths), 1)

    df = Counter()

    for tokens in corpus:
        df.update(set(tokens))

    scores = np.zeros(len(corpus), dtype=float)
    n = len(corpus)

    for i, tokens in enumerate(corpus):
        tf = Counter(tokens)

        for term in query_terms:
            freq = tf[term]

            if not freq:
                continue

            idf = math.log(
                1 + (n - df[term] + 0.5) / (df[term] + 0.5)
            )

            denom = freq + k1 * (
                1 - b + b * lengths[i] / max(avg_len, 1)
            )

            scores[i] += idf * freq * (k1 + 1) / denom

    maximum = float(scores.max()) if scores.size else 0.0

    return scores / maximum if maximum > 0 else scores


def _path_matches(path: str, pattern: str) -> bool:
    from fnmatch import fnmatch

    pattern = pattern.replace("\\", "/").strip()

    if pattern.endswith("/"):
        return path.startswith(pattern)

    return fnmatch(path, pattern) or fnmatch(path, f"**/{pattern}")


def search_functions(
    query,
    documents,
    embeddings,
    model,
    top_k=5,
    semantic_weight=0.75,
    lexical_weight=0.25,
    file_filter=None,
    symbol_type=None,
    search_mode="hybrid"
):
    """Rank code using optimized semantic, lexical, or hybrid search."""

    if search_mode not in ("semantic", "lexical", "hybrid"):
        raise ValueError(
            "search_mode must be 'semantic', 'lexical', or 'hybrid'."
        )

    if not 0 <= semantic_weight <= 1 or not 0 <= lexical_weight <= 1:
        raise ValueError("Weights must be between 0 and 1.")

    if semantic_weight + lexical_weight == 0:
        raise ValueError("At least one weight must be greater than zero.")

    if not documents:
        return []

    needs_semantic = search_mode in ("semantic", "hybrid")
    needs_lexical = search_mode in ("lexical", "hybrid")

    # Calculate semantic scores only when required.
    if needs_semantic:
        if embeddings is None or len(embeddings) != len(documents):
            raise ValueError(
                "Embedding count must match the number of code documents."
            )

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]

        cosine = np.asarray(embeddings) @ query_embedding
        semantic = np.clip((cosine + 1) / 2, 0, 1)
    else:
        semantic = np.zeros(len(documents), dtype=float)

    # Calculate lexical scores only when required.
    if needs_lexical:
        lexical = bm25_scores(query, documents)
    else:
        lexical = np.zeros(len(documents), dtype=float)

    # Determine final ranking scores.
    if search_mode == "semantic":
        combined = semantic

    elif search_mode == "lexical":
        combined = lexical

    else:
        weight_sum = semantic_weight + lexical_weight

        combined = (
            semantic_weight * semantic
            + lexical_weight * lexical
        ) / weight_sum

    results = []
    query_terms = set(tokenize(query))

    for i, doc in enumerate(documents):

        if file_filter and not _path_matches(
            doc["file_path"], file_filter
        ):
            continue

        if symbol_type and doc["symbol_type"] != symbol_type:
            continue

        code_terms = set(
            tokenize(
                f"{doc['qualified_name']} {doc['signature']} "
                f"{doc['docstring']} {doc['source']}"
            )
        )

        matches = sorted(query_terms & code_terms)
        reasons = []

        if needs_semantic and semantic[i] >= 0.60:
            reasons.append(
                "the code is semantically similar to the query"
            )

        if matches:
            reasons.append(
                "matching terms: " + ",".join(matches[:8])
            )

        if not reasons:
            reasons.append("strongest available combined ranking")

        results.append({
            "document": doc,
            "score": float(combined[i]),
            "semantic_score": float(semantic[i]),
            "lexical_score": float(lexical[i]),
            "explanation": "; ".join(reasons)
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:max(1, int(top_k))]
