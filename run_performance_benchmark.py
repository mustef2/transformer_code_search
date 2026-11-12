
import argparse
import csv
import json
import statistics
import time

from pathlib import Path

from code_search.database import PersistentCodeIndex
from code_search.search import search_functions

from run_evaluation import (
    SEARCH_STRATEGIES,
    load_model,
    load_benchmark,
)


def benchmark_strategy(
    documents,
    embeddings,
    model,
    queries,
    strategy,
    repeats,
    warmup,
    top_k,
):
    """Measure execution time for one search strategy."""

    print(f"\nBenchmarking {strategy['name'].upper()}...")

    all_times = []
    query_details = []

    for query_item in queries:

        query = query_item["query"]

        search_arguments = {
            "query": query,
            "documents": documents,
            "embeddings": embeddings,
            "model": model,
            "top_k": top_k,
            "semantic_weight": strategy["semantic_weight"],
            "lexical_weight": strategy["lexical_weight"],
            "search_mode": strategy["mode"],
        }

        # Warm-up runs are excluded from measurements.
        for _ in range(warmup):
            search_functions(**search_arguments)

        query_times = []

        # Measure repeated searches.
        for _ in range(repeats):

            start_time = time.perf_counter()

            search_functions(**search_arguments)

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000

            query_times.append(elapsed_ms)
            all_times.append(elapsed_ms)

        query_details.append({
            "query": query,
            "average_ms": statistics.mean(query_times),
            "median_ms": statistics.median(query_times),
            "minimum_ms": min(query_times),
            "maximum_ms": max(query_times),
            "standard_deviation_ms": (
                statistics.stdev(query_times)
                if len(query_times) > 1
                else 0.0
            ),
        })

    return {
        "mode": strategy["mode"],
        "semantic_weight": strategy["semantic_weight"],
        "lexical_weight": strategy["lexical_weight"],
        "average_ms": statistics.mean(all_times),
        "median_ms": statistics.median(all_times),
        "minimum_ms": min(all_times),
        "maximum_ms": max(all_times),
        "standard_deviation_ms": (
            statistics.stdev(all_times)
            if len(all_times) > 1
            else 0.0
        ),
        "total_measurements": len(all_times),
        "total_time_ms": sum(all_times),
        "queries": query_details,
    }


def run_benchmark(
    documents,
    embeddings,
    model,
    queries,
    repeats,
    warmup,
    top_k,
):
    """Benchmark all seven search strategies."""

    results = {}

    for strategy in SEARCH_STRATEGIES:

        results[strategy["name"]] = benchmark_strategy(
            documents=documents,
            embeddings=embeddings,
            model=model,
            queries=queries,
            strategy=strategy,
            repeats=repeats,
            warmup=warmup,
            top_k=top_k,
        )

    return results


def print_comparison(results):

    print("\n" + "=" * 105)
    print("SEARCH STRATEGY PERFORMANCE COMPARISON")
    print("=" * 105)

    print(
        f"{'Strategy':<20}"
        f"{'Average (ms)':>15}"
        f"{'Median (ms)':>15}"
        f"{'Minimum (ms)':>15}"
        f"{'Maximum (ms)':>15}"
        f"{'Std Dev (ms)':>15}"
    )

    print("-" * 105)

    for name, data in results.items():

        print(
            f"{name:<20}"
            f"{data['average_ms']:>15.3f}"
            f"{data['median_ms']:>15.3f}"
            f"{data['minimum_ms']:>15.3f}"
            f"{data['maximum_ms']:>15.3f}"
            f"{data['standard_deviation_ms']:>15.3f}"
        )

    print("=" * 105)

    fastest = min(
        results,
        key=lambda name: results[name]["average_ms"]
    )

    print("\nFASTEST SEARCH STRATEGY")
    print("-" * 50)

    print(
        f"{fastest}: "
        f"{results[fastest]['average_ms']:.3f} ms average"
    )

    print("-" * 50)


def save_reports(results, output_dir, repeats, warmup, top_k, query_count):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = output_dir / "performance_results.csv"
    json_path = output_dir / "performance_results.json"

    # Save summary CSV.
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Strategy",
            "Search Mode",
            "Semantic Weight",
            "Lexical Weight",
            "Average (ms)",
            "Median (ms)",
            "Minimum (ms)",
            "Maximum (ms)",
            "Standard Deviation (ms)",
            "Total Measurements",
            "Total Time (ms)",
        ])

        for name, data in results.items():

            writer.writerow([
                name,
                data["mode"],
                data["semantic_weight"],
                data["lexical_weight"],
                f"{data['average_ms']:.4f}",
                f"{data['median_ms']:.4f}",
                f"{data['minimum_ms']:.4f}",
                f"{data['maximum_ms']:.4f}",
                f"{data['standard_deviation_ms']:.4f}",
                data["total_measurements"],
                f"{data['total_time_ms']:.4f}",
            ])

    # Save detailed JSON.
    report = {
        "benchmark_configuration": {
            "query_count": query_count,
            "repeats_per_query": repeats,
            "warmup_runs_per_query": warmup,
            "top_k": top_k,
            "total_strategies": len(results),
        },
        "results": results,
    }

    json_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print(f"\nCSV report saved: {csv_path}")
    print(f"JSON report saved: {json_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Benchmark Transformer-based code search performance."
    )

    parser.add_argument(
        "--repo",
        default=".",
        help="Python repository to benchmark."
    )

    parser.add_argument(
        "--db",
        default="data/code_search.db",
        help="SQLite database path."
    )

    parser.add_argument(
        "--benchmark",
        default="evaluation_queries.json",
        help="Benchmark queries JSON file."
    )

    parser.add_argument(
        "--output",
        default="performance_results",
        help="Output directory for performance reports."
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Measured repetitions per query and strategy."
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up runs per query and strategy."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results to retrieve."
    )

    args = parser.parse_args()

    if args.repeats <= 0:
        parser.error("--repeats must be greater than zero.")

    if args.warmup < 0:
        parser.error("--warmup cannot be negative.")

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero.")

    repo_path = Path(args.repo).expanduser().resolve()
    database_path = Path(args.db).expanduser().resolve()
    benchmark_path = Path(args.benchmark).expanduser().resolve()

    if not repo_path.is_dir():
        parser.error(f"Repository folder not found: {repo_path}")

    if not benchmark_path.is_file():
        parser.error(f"Benchmark file not found: {benchmark_path}")

    print("Loading Transformer model...")

    model = load_model()

    print("\nPreparing persistent code index...")

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
        raise SystemExit("No Python code units found.")

    queries = load_benchmark(benchmark_path)

    print("\nPERFORMANCE BENCHMARK CONFIGURATION")
    print("-" * 50)
    print(f"Benchmark queries: {len(queries)}")
    print(f"Search strategies: {len(SEARCH_STRATEGIES)}")
    print(f"Warm-up runs: {args.warmup}")
    print(f"Measured repetitions: {args.repeats}")
    print(f"Top K: {args.top_k}")
    print(
        "Total measured searches:",
        len(queries) * len(SEARCH_STRATEGIES) * args.repeats
    )
    print("-" * 50)

    results = run_benchmark(
        documents=documents,
        embeddings=embeddings,
        model=model,
        queries=queries,
        repeats=args.repeats,
        warmup=args.warmup,
        top_k=args.top_k,
    )

    print_comparison(results)

    save_reports(
        results=results,
        output_dir=Path(args.output),
        repeats=args.repeats,
        warmup=args.warmup,
        top_k=args.top_k,
        query_count=len(queries),
    )


if __name__ == "__main__":
    main()
