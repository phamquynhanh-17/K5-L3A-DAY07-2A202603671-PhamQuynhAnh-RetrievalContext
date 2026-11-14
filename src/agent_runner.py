"""Agentic RAG runner using Gemini (gemini-3.8-flash) and Day 6/7 retrievers."""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
from agentic import run_agent  # noqa: E402
from check_gemini import gemini_generate  # noqa: E402
from retrievers import Day6Retriever  # noqa: E402


SYSTEM_PROMPT = """Bạn là trợ lý hỏi-đáp dựa trên bằng chứng từ tài liệu.
Quy tắc bắt buộc:
1. KHÔNG BAO GIỜ dùng kiến thức ngoài tài liệu hoặc suy đoán. Mọi khẳng định phải có bằng chứng trực tiếp từ các đoạn văn bản (chunk) đã truy xuất.
2. Nếu chưa có thông tin hoặc các đoạn đã tìm mới chỉ nhắc thoáng qua tên chương/khái niệm mà chưa có nội dung trả lời câu hỏi, hãy gọi action "search" với query khác rõ từ khóa hơn.
3. Khi đã có đủ bằng chứng trực tiếp trong context, trả về action "answer", trong đó:
   - "answer": câu trả lời chính xác, ngắn gọn, kèm trích dẫn dạng [chunk_id] ngay sau thông tin tương ứng.
   - "citations": danh sách các chunk_id đã trích dẫn (phải xuất hiện đúng dạng [chunk_id] trong answer và nằm trong context_ids).
4. Nếu đã tìm kiếm nhưng tài liệu KHÔNG chứa đủ bằng chứng để trả lời câu hỏi trọng tâm (kể cả khi chỉ nhắc tên chương mà không giải thích nội dung được hỏi), BẮT BUỘC từ chối bằng đúng JSON sau (tuyệt đối để "citations" là mảng rỗng []):
   {"action": "answer", "answer": "Không đủ bằng chứng trong tài liệu.", "citations": []}
5. Chỉ trả về duy nhất một JSON object hợp lệ, không kèm markdown."""


class GeminiController:
    """Real LLM controller using Gemini 3.1 Flash Lite via REST API."""

    def __init__(self, model_name: str = "gemini-3.1-flash-lite") -> None:
        self.model_name = model_name
        self.total_tokens = 0

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            user_content = (
                f"Câu hỏi: {question}\n\n"
                "Chưa có lượt tìm kiếm nào. Hãy tạo truy vấn tìm kiếm đầu tiên dưới dạng JSON: "
                '{"action": "search", "query": "..."}'
            )
        else:
            last = history[-1]
            queries_str = ", ".join(repr(q) for q in last.get("queries", []))
            ctx_ids = last.get("context_ids", [])
            ctx_text = last.get("context", "") or "(Không tìm thấy đoạn nào phù hợp)"
            user_content = (
                f"Câu hỏi: {question}\n"
                f"Các truy vấn đã chạy: [{queries_str}]\n"
                f"Các chunk_id hợp lệ trong context: {json.dumps(ctx_ids, ensure_ascii=False)}\n\n"
                f"Context hiện tại:\n{ctx_text}\n\n"
                "Hãy quyết định: nếu cần tìm thêm (ví dụ thiếu thông tin hoặc cần từ khóa khác), trả về "
                '{"action": "search", "query": "..."}. '
                "Nếu đã đủ bằng chứng để trả lời có trích dẫn [chunk_id], hoặc chắc chắn không có trong tài liệu, trả về "
                '{"action": "answer", "answer": "...", "citations": [...]}.'
            )

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                text, tokens = gemini_generate(
                    user_content,
                    model=self.model_name,
                    system_instruction=SYSTEM_PROMPT,
                    json_mode=True,
                    timeout=25,
                )
                self.total_tokens += tokens
                data = json.loads(text or "{}")
                if data.get("action") in ("search", "answer"):
                    return data
                raise ValueError(f"Invalid action JSON: {data}")
            except Exception as exc:
                last_err = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"GeminiController failed: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=("tfidf", "dense", "hybrid_rrf", "hybrid_rerank"))
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=2400)
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data_dir = ROOT / "starter" / "data" / "day6_c"
    chunks = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
    vectors = json.loads((data_dir / "vectors.json").read_text(encoding="utf-8"))["vectors"]
    cases = [
        c for c in json.loads((data_dir / "agent_cases.json").read_text(encoding="utf-8"))
        if c["split"] == args.split
    ]

    retriever = Day6Retriever(
        chunks=chunks,
        vectors=vectors,
        strategy=args.strategy,
        query_cache_path=data_dir / "query_emb_cache.json",
    )
    controller = GeminiController()

    traces = []
    for case in cases:
        t0 = time.perf_counter()
        tok_before = controller.total_tokens + retriever.rerank_tokens
        try:
            res = run_agent(
                question=case["question"],
                controller=controller,
                retriever=retriever,
                filters=case.get("filter"),
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            tokens = (controller.total_tokens + retriever.rerank_tokens) - tok_before
            traces.append(
                {
                    "query_id": case["id"],
                    "split": case["split"],
                    "answerable": case["answerable"],
                    "question": case["question"],
                    "status": res["status"],
                    "answer": res["answer"],
                    "citations": res["citations"],
                    "events": res["events"],
                    "elapsed_ms": elapsed_ms,
                    "tokens": tokens,
                    "error": None,
                }
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            traces.append(
                {
                    "query_id": case["id"],
                    "split": case["split"],
                    "answerable": case["answerable"],
                    "question": case["question"],
                    "status": "error",
                    "answer": "",
                    "citations": [],
                    "events": [],
                    "elapsed_ms": elapsed_ms,
                    "tokens": 0,
                    "error": str(exc),
                }
            )
        print(f"[{case['id']}] status={traces[-1]['status']} cites={traces[-1]['citations']} ms={traces[-1]['elapsed_ms']}", flush=True)
        time.sleep(0.5)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "strategy": args.strategy,
        "split": args.split,
        "top_k": args.top_k,
        "budget": args.budget,
        "max_searches": args.max_searches,
        "model": controller.model_name,
        "total_tokens": controller.total_tokens + retriever.rerank_tokens,
        "total_cases": len(traces),
        "errors": sum(1 for t in traces if t["status"] == "error"),
    }
    (args.output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

