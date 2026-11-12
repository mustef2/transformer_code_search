
import argparse
import csv
import json
from pathlib import Path

from code_search.database import PersistentCodeIndex
from code_search.search import search_functions
from code_search.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# Seven search strategies to compare.
SEARCH_STRATEGIES = [
    {
        "name": "semantic",
        "mode": "semantic",
        "semantic_weight": 1.0,
        "lexical_weight": 0.0,
    },
    {
        "name": "lexical",
        "mode": "lexical",
        "semantic_weight": 0.0,
        "lexical_weight": 1.0,
    },
    {
        "name": "hybrid_90_10",
        "mode": "hybrid",
        "semantic_weight": 0.90,
        "lexical_weight": 0.10,
    },
    {
        "name": "hybrid_85_15",
        "mode": "hybrid",
        "semantic_weight": 0.85,
        "lexical_weight": 0.15,
    },
    {
        "name": "hybrid_80_20",
        "mode": "hybrid",
        "semantic_weight": 0.80,
        "lexical_weight": 0.20,
    },
    {
        "name": "hybrid_75_25",
        "mode": "hybrid",
        "semantic_weight": 0.75,
        "lexical_weight": 0.25,
    },
    {
        "name": "hybrid_70_30",
        "mode": "hybrid",
        "semantic_weight": 0.70,
        "lexical_weight": 0.30,
    },
]


def document_id(document):
    """Create a stable identifier for an extracted code unit."""

    return (
        f"{document['file_path']}::"
        f"{document['qualified_name']}"
    )


def load_model():
    """Load the same Transformer model used by the application."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def create_catalog(documents, output_path):
    """Export indexed code units for manual relevance labeling."""

    catalog = []

    for document in documents:
        catalog.append({
            "id": document_id(document),
            "file_path": document["file_path"],
            "qualified_name": document["qualified_name"],
            "signature": document["signature"],
            "symbol_type": document["symbol_type"],
            "docstring": document["docstring"],
        })

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        json.dumps(catalog, indent=4),
        encoding="utf-8"
    )

    print(f"\nCatalog created: {output_path}")
    print(f"Total code units: {len(catalog)}")

    print("\nFirst 10 code units:")

    for item in catalog[:10]:
        print(f"\nID: {item['id']}")
        print(f"Signature: {item['signature']}")
        print(f"Docstring: {item['docstring'][:200]}")


def load_benchmark(benchmark_path):
    """Load and validate benchmark queries."""

    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {benchmark_path}"
        )

    data = json.loads(
        benchmark_path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            "Benchmark must contain a JSON object."
        )

    queries = data.get("queries")

    if not isinstance(queries, list) or not queries:
        raise ValueError(
            "Benchmark must contain a non-empty 'queries' list."
        )

    for index, item in enumerate(queries, start=1):

        if not isinstance(item, dict):
            raise ValueError(
                f"Benchmark query {index} must be an object."
            )

        query = item.get("query")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Benchmark query {index} has no valid query text."
            )

        relevant = item.get("relevant_ids")

        if not isinstance(relevant, list) or not relevant:
            raise ValueError(
                f"Query {index} needs at least one manually verified "
                "relevant ID."
            )

        if not all(
            isinstance(identifier, str) and identifier.strip()
            for identifier in relevant
        ):
            raise ValueError(
                f"Query {index} contains an invalid relevant ID."
            )

    return queries


def validate_relevant_ids(documents, queries):
    """Ensure all relevance IDs exist in the indexed repository."""

    known_ids = {
        document_id(document)
        for document in documents
    }

    for item in queries:

        relevant = set(item["relevant_ids"])
        unknown_ids = relevant - known_ids

        if unknown_ids:
            raise ValueError(
                f"Unknown relevant IDs for query '{item['query']}': "
                f"{sorted(unknown_ids)}"
            )


def evaluate_strategy(
    documents,
    embeddings,
    model,
    queries,
    strategy,
    k,
):
    """Evaluate one search strategy across all benchmark queries."""

    print(f"\nEvaluating {strategy['name'].upper()} search...")

    query_results = []

    for item in queries:

        query = item["query"]
        relevant = set(item["relevant_ids"])

        results = search_functions(
            query=query,
            documents=documents,
            embeddings=embeddings,
            model=model,
            top_k=k,
            semantic_weight=strategy["semantic_weight"],
            lexical_weight=strategy["lexical_weight"],
            search_mode=strategy["mode"],
        )

        retrieved = [
            document_id(result["document"])
            for result in results
        ]

        query_results.append({
            "query": query,

            "precision_at_k": precision_at_k(
                retrieved, relevant, k
            ),

            "recall_at_k": recall_at_k(
                retrieved, relevant, k
            ),

            "reciprocal_rank": reciprocal_rank(
                retrieved, relevant
            ),

            "retrieved_ids": retrieved,
            "relevant_ids": sorted(relevant),
        })

    count = len(query_results)

    return {
        "mode": strategy["mode"],
        "semantic_weight": strategy["semantic_weight"],
        "lexical_weight": strategy["lexical_weight"],

        "precision_at_k": sum(
            item["precision_at_k"]
            for item in query_results
        ) / count,

        "recall_at_k": sum(
            item["recall_at_k"]
            for item in query_results
        ) / count,

        "mrr": sum(
            item["reciprocal_rank"]
            for item in query_results
        ) / count,

        "queries": query_results,
    }


def run_evaluation(
    documents,
    embeddings,
    model,
    queries,
    k=5,
):
    """Evaluate all seven search strategies."""

    validate_relevant_ids(documents, queries)

    results_by_strategy = {}

    for strategy in SEARCH_STRATEGIES:

        results_by_strategy[strategy["name"]] = evaluate_strategy(
            documents=documents,
            embeddings=embeddings,
            model=model,
            queries=queries,
            strategy=strategy,
            k=k,
        )

    return results_by_strategy


def print_comparison(results, k):
    """Print comparison of all seven search strategies."""

    print("\n" + "=" * 85)
    print("SEARCH STRATEGY AND HYBRID WEIGHT COMPARISON")
    print("=" * 85)

    print(
        f"{'Strategy':<20}"
        f"{'Semantic':>11}"
        f"{'Lexical':>11}"
        f"{'Precision@' + str(k):>14}"
        f"{'Recall@' + str(k):>12}"
        f"{'MRR':>10}"
    )

    print("-" * 85)

    for name, data in results.items():

        print(
            f"{name:<20}"
            f"{data['semantic_weight']:>10.0%}"
            f"{data['lexical_weight']:>10.0%}"
            f"{data['precision_at_k']:>14.4f}"
            f"{data['recall_at_k']:>12.4f}"
            f"{data['mrr']:>10.4f}"
        )

    print("=" * 85)

    # Find the best strategy for each metric.
    metrics = [
        ("precision_at_k", "Precision"),
        ("recall_at_k", "Recall"),
        ("mrr", "MRR"),
    ]

    print("\nBEST CONFIGURATION BY METRIC")
    print("-" * 55)

    for metric, label in metrics:

        best_name = max(
            results,
            key=lambda name: results[name][metric]
        )

        best_score = results[best_name][metric]

        print(
            f"{label:<15}: {best_name:<20} "
            f"({best_score:.4f})"
        )

    print("-" * 55)


def save_reports(results, output_dir, k):
    """Save detailed CSV and JSON evaluation reports."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = output_dir / "evaluation_results.csv"
    json_path = output_dir / "evaluation_results.json"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Search Strategy",
            "Search Mode",
            "Semantic Weight",
            "Lexical Weight",
            "Query",
            f"Precision@{k}",
            f"Recall@{k}",
            "Reciprocal Rank",
            "Retrieved IDs",
            "Relevant IDs",
        ])

        for strategy, data in results.items():

            for item in data["queries"]:

                writer.writerow([
                    strategy,
                    data["mode"],
                    data["semantic_weight"],
                    data["lexical_weight"],
                    item["query"],
                    f"{item['precision_at_k']:.4f}",
                    f"{item['recall_at_k']:.4f}",
                    f"{item['reciprocal_rank']:.4f}",
                    " | ".join(item["retrieved_ids"]),
                    " | ".join(item["relevant_ids"]),
                ])

    report_data = {
        "k": k,

        "summary": {
            strategy: {
                "mode": data["mode"],
                "semantic_weight": data["semantic_weight"],
                "lexical_weight": data["lexical_weight"],
                "precision_at_k": data["precision_at_k"],
                "recall_at_k": data["recall_at_k"],
                "mrr": data["mrr"],
            }
            for strategy, data in results.items()
        },

        "details": results,
    }

    json_path.write_text(
        json.dumps(report_data, indent=4),
        encoding="utf-8"
    )

    print(f"\nCSV report saved: {csv_path}")
    print(f"JSON report saved: {json_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate Transformer-based code search."
    )

    parser.add_argument(
        "--repo",
        default=".",
        help="Path to the Python repository to evaluate."
    )

    parser.add_argument(
        "--db",
        default="data/code_search.db",
        help="SQLite database path."
    )

    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Generate the code catalog without running evaluation."
    )

    parser.add_argument(
        "--benchmark",
        default="evaluation_queries.json",
        help="Path to the labeled benchmark JSON file."
    )

    parser.add_argument(
        "--output",
        default="evaluation_results",
        help="Directory for evaluation reports."
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top results used for Precision@K and Recall@K."
    )

    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be greater than zero.")

    repo_path = Path(args.repo).expanduser().resolve()
    database_path = Path(args.db).expanduser().resolve()

    if not repo_path.is_dir():
        parser.error(
            f"Repository folder does not exist: {repo_path}"
        )

    print("Loading Transformer model...")

    model = load_model()

    print("Preparing persistent code index...")

    index = PersistentCodeIndex(database_path)

    documents, embeddings, stats = index.index_repository(
        repo_path,
        model
    )

    print("\nIndex statistics:")
    print(f"Python files: {stats['total_files']}")
    print(f"Code units: {stats['indexed_code_units']}")
    print(f"Changed files: {stats['changed_files']}")
    print(f"Reused files: {stats['reused_files']}")

    if not documents:
        raise SystemExit(
            "No Python code units were found. Nothing to evaluate."
        )

    if args.catalog_only:

        create_catalog(
            documents,
            Path("evaluation_catalog.json")
        )

        return

    queries = load_benchmark(
        Path(args.benchmark)
    )

    print(f"\nBenchmark queries: {len(queries)}")
    print(f"Evaluation cutoff: {args.k}")
    print(f"Search strategies: {len(SEARCH_STRATEGIES)}")

    results = run_evaluation(
        documents=documents,
        embeddings=embeddings,
        model=model,
        queries=queries,
        k=args.k,
    )

    print_comparison(results, args.k)

    save_reports(
        results,
        Path(args.output),
        args.k
    )


if __name__ == "__main__":
    main()
