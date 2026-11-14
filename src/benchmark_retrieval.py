"""Run retrieval benchmarks across strategies and budgets on dev/test splits."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
from baseline import hit, pack_context  # noqa: E402
from retrievers import Day6Retriever  # noqa: E402


def evaluate_retriever(
    retriever: Day6Retriever,
    questions: list[dict],
    top_k: int = 3,
    budget: int = 800,
    use_filter: bool = True,
) -> tuple[dict, list[dict]]:
    traces = []
    t0 = time.perf_counter()
    for q in questions:
        q_filter = q.get("filter") if use_filter else None
        start = time.perf_counter()
        results = retriever.search(q["question"], filters=q_filter, top_k=top_k)
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        marked = [{**c, "hit": hit(c, q["gold"])} for c in results]
        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "question": q["question"],
                "filter": q_filter,
                "strategy": retriever.strategy,
                "top_k": top_k,
                "budget_chars": budget,
                "gold": q["gold"],
                "results": marked,
                "context": context,
                "context_ids": ids,
                "context_chars": len(context),
                "elapsed_ms": round(elapsed_ms, 2),
                "hit1": bool(marked and marked[0]["hit"]),
                "hit3": any(c["hit"] for c in marked[:3]),
                "context_hit": any(c["hit"] and c["id"] in ids for c in marked),
            }
        )
    total_s = time.perf_counter() - t0
    n = len(traces)
    h1 = sum(t["hit1"] for t in traces)
    h3 = sum(t["hit3"] for t in traces)
    ch = sum(t["context_hit"] for t in traces)
    summary = {
        "strategy": retriever.strategy,
        "top_k": top_k,
        "budget_chars": budget,
        "use_filter": use_filter,
        "n": n,
        "hit1": h1,
        "hit1_rate": round(h1 / n, 4) if n else 0.0,
        "hit3": h3,
        "hit3_rate": round(h3 / n, 4) if n else 0.0,
        "context_hit": ch,
        "context_rate": round(ch / n, 4) if n else 0.0,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1) if n else 0.0,
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2) if n else 0.0,
        "total_seconds": round(total_s, 3),
        "rerank_tokens": retriever.rerank_tokens,
        "rerank_errors": retriever.rerank_errors,
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=("tfidf", "dense", "hybrid_rrf", "hybrid_rerank"))
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data_dir = ROOT / "starter" / "data" / "day6_c"
    chunks = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
    vectors = json.loads((data_dir / "vectors.json").read_text(encoding="utf-8"))["vectors"]
    questions = [
        q for q in json.loads((data_dir / "questions.json").read_text(encoding="utf-8"))
        if q["split"] == args.split
    ]

    retriever = Day6Retriever(
        chunks=chunks,
        vectors=vectors,
        strategy=args.strategy,
        query_cache_path=data_dir / "query_emb_cache.json",
    )
    summary, traces = evaluate_retriever(
        retriever,
        questions,
        top_k=args.top_k,
        budget=args.budget,
        use_filter=not args.no_filter,
    )
    summary["split"] = args.split
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

