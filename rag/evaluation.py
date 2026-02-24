"""Evaluation module: test set runner and retrieval metrics."""

import json

from rich.console import Console
from rich.table import Table

from rag import config
from rag.retriever import retrieve

console = Console()


def load_test_set(path=None) -> list[dict]:
    """Load the evaluation test set from JSON."""
    if path is None:
        path = config.EVAL_DIR / "test_set.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int) -> float:
    """Fraction of expected sources found in the top-k retrieved sources."""
    if not expected_sources:
        return 1.0
    retrieved_set = set(retrieved_sources[:k])
    hits = sum(1 for s in expected_sources if s in retrieved_set)
    return hits / len(expected_sources)


def precision_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int) -> float:
    """Fraction of top-k retrieved sources that are in the expected set."""
    top_k = retrieved_sources[:k]
    if not top_k:
        return 0.0
    expected_set = set(expected_sources)
    hits = sum(1 for s in top_k if s in expected_set)
    return hits / len(top_k)


def mrr(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant result."""
    expected_set = set(expected_sources)
    for i, source in enumerate(retrieved_sources, start=1):
        if source in expected_set:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_retrieval_eval(top_k: int = config.TOP_K):
    """Run all test questions through retrieval and report metrics."""
    test_set = load_test_set()

    if not test_set:
        console.print("[red]No test questions found.[/red]")
        return

    console.print(f"Evaluating {len(test_set)} questions (top_k={top_k})\n")

    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0

    results_table = Table(title="Retrieval Evaluation Results")
    results_table.add_column("#", style="dim", width=4)
    results_table.add_column("Difficulty", width=8)
    results_table.add_column("Question", max_width=50)
    results_table.add_column("Recall@k", justify="right", width=10)
    results_table.add_column("Prec@k", justify="right", width=10)
    results_table.add_column("MRR", justify="right", width=10)

    for idx, item in enumerate(test_set, start=1):
        question = item["question"]
        expected = item.get("expected_sources", [])

        chunks = retrieve(question, top_k=top_k)
        retrieved_sources = [c["source"] for c in chunks]

        r = recall_at_k(retrieved_sources, expected, top_k)
        p = precision_at_k(retrieved_sources, expected, top_k)
        m = mrr(retrieved_sources, expected)

        total_recall += r
        total_precision += p
        total_mrr += m

        difficulty = item.get("difficulty", "?")
        results_table.add_row(
            str(idx),
            difficulty,
            question[:50],
            f"{r:.2f}",
            f"{p:.2f}",
            f"{m:.2f}",
        )

    console.print(results_table)

    n = len(test_set)
    console.print(f"\n[bold]Aggregate Metrics (n={n}):[/bold]")
    console.print(f"  Mean Recall@{top_k}:    {total_recall / n:.3f}")
    console.print(f"  Mean Precision@{top_k}: {total_precision / n:.3f}")
    console.print(f"  Mean MRR:              {total_mrr / n:.3f}")
