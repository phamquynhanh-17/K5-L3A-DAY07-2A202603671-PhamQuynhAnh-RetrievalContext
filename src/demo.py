"""Interactive / live demo for Day 7 Agentic RAG:
1. Answered case with grounded citation (live run)
2. Unanswerable case with abstention (live run)
3. Error analysis comparison (Baseline vs Chosen config)
Supports custom question via: python src/demo.py --question "..."
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "src"))

from agentic import load_real_adapters, run_agent  # noqa: E402


def print_trace_box(title: str, question: str, res: dict, elapsed_ms: float, tokens: int) -> None:
    print("=" * 80)
    print(f"DEMO: {title}")
    print("=" * 80)
    print(f"Câu hỏi : {question}")
    print(f"Trạng thái: {res['status'].upper()} | Thời gian: {elapsed_ms:.1f} ms | Tokens: {tokens}")
    searches = [e for e in res["events"] if e.get("event") == "search"]
    print(f"Số lượt gọi search tool: {len(searches)}")
    for idx, s in enumerate(searches, start=1):
        top_ids = [f"{r['id']} (p.{r['page']}, score={r['score']})" for r in s["results"]]
        print(f"  [Lượt {idx}] Query: {s['query']!r}")
        print(f"           Top-3 retrieved: {top_ids}")
        print(f"           Context IDs (in budget): {s['context_ids']}")
    print("-" * 80)
    print(f"Câu trả lời: {res['answer']}")
    print(f"Citations  : {res['citations']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy demo trực tiếp Agentic RAG Day 7")
    parser.add_argument("--question", type=str, default=None, help="Câu hỏi tùy chọn để chạy thử trực tiếp")
    parser.add_argument("--strategy", default="hybrid_rerank", choices=("tfidf", "dense", "hybrid_rrf", "hybrid_rerank"))
    parser.add_argument("--budget", type=int, default=3600)
    parser.add_argument("--max-searches", type=int, default=4)
    args = parser.parse_args()

    controller, retriever = load_real_adapters(strategy=args.strategy)

    if args.question:
        t0 = time.perf_counter()
        tok0 = controller.total_tokens + retriever.rerank_tokens
        res = run_agent(
            args.question,
            controller,
            retriever,
            max_searches=args.max_searches,
            top_k=3,
            budget=args.budget,
        )
        ms = (time.perf_counter() - t0) * 1000
        tok = (controller.total_tokens + retriever.rerank_tokens) - tok0
        print_trace_box(f"Custom Question ({args.strategy}, budget={args.budget})", args.question, res, ms, tok)
        return

    # 1. Live run: Answerable question with grounded citation
    q_ans = "Theo lời nói đầu của Warren E. Buffett, hai chương nào trong sách cần được đặc biệt chú ý tới?"
    t0 = time.perf_counter()
    tok0 = controller.total_tokens + retriever.rerank_tokens
    res_ans = run_agent(q_ans, controller, retriever, max_searches=4, top_k=3, budget=3600)
    ms_ans = (time.perf_counter() - t0) * 1000
    tok_ans = (controller.total_tokens + retriever.rerank_tokens) - tok0
    print_trace_box("1. Câu hỏi CÓ BẰNG CHỨNG (hybrid_rerank, budget=3600)", q_ans, res_ans, ms_ans, tok_ans)

    # 2. Live run: Unanswerable question -> Abstention
    q_abs = "Trong Chương 8 của cuốn sách, Benjamin Graham giải thích khái niệm 'Ngài Thị trường' (Mr. Market) như thế nào?"
    t0 = time.perf_counter()
    tok0 = controller.total_tokens + retriever.rerank_tokens
    res_abs = run_agent(q_abs, controller, retriever, max_searches=4, top_k=3, budget=3600)
    ms_abs = (time.perf_counter() - t0) * 1000
    tok_abs = (controller.total_tokens + retriever.rerank_tokens) - tok0
    print_trace_box("2. Câu hỏi THIẾU BẰNG CHỨNG — Kiểm tra Abstention (hybrid_rerank, budget=3600)", q_abs, res_abs, ms_abs, tok_abs)

    # 3. Error Case Comparison: Baseline (tfidf, budget=800) vs Chosen on q20 (Test)
    print("=" * 80)
    print("DEMO: 3. Minh họa LỖI & CẢI THIỆN trên tập Test (Câu q20)")
    print("=" * 80)
    base_traces = json.loads((ROOT / "starter" / "results" / "test-agent-baseline" / "traces.json").read_text(encoding="utf-8"))
    chosen_traces = json.loads((ROOT / "starter" / "results" / "test-agent-chosen" / "traces.json").read_text(encoding="utf-8"))
    b_q20 = next(t for t in base_traces if t["query_id"] == "q20")
    c_q20 = next(t for t in chosen_traces if t["query_id"] == "q20")
    print(f"Câu hỏi q20: {b_q20['question']}")
    print(f"[Baseline tfidf, budget=800]")
    print(f"  - Top-3 retrieved: {[r['id'] for r in b_q20['events'][0]['results']]}")
    print(f"  - Context IDs (sau budget=800): {b_q20['events'][0]['context_ids']}  <-- Không lấy được p14_para_1 do TF-IDF + budget=800!")
    print(f"  - Kết quả: status={b_q20['status']} | answer={b_q20['answer']!r}")
    print(f"[Cấu hình chọn hybrid_rerank, budget=3600]")
    print(f"  - Top-3 retrieved: {[r['id'] for r in c_q20['events'][0]['results']]}")
    print(f"  - Context IDs (sau budget=3600): {c_q20['events'][0]['context_ids']} <-- Giữ trọn p14_para_1!")
    print(f"  - Kết quả: status={c_q20['status']} | citations={c_q20['citations']}")
    print(f"  - Câu trả lời: {c_q20['answer']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
