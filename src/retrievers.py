"""Retrieval methods for Day 7: TF-IDF, Dense (Day 6 3072-dim), Hybrid RRF, and Hybrid + LLM Rerank."""
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
from baseline import pack_context  # noqa: E402
from check_gemini import gemini_generate, load_gemini_key  # noqa: E402


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def matches_filter(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    meta = chunk.get("metadata", {})
    for k, v in filters.items():
        if k in ("source_id", "source", "page"):
            if chunk.get(k) != v:
                return False
        elif meta.get(k) != v:
            return False
    return True


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class GeminiQueryEmbedder:
    """Query embedder matching Day 6 (gemini-embedding-001, 3072 dims, RETRIEVAL_QUERY)."""

    def __init__(self, cache_path: Path | None = None) -> None:
        self.api_key = load_gemini_key()
        self.model_name = "gemini-embedding-001"
        self.cache_path = cache_path
        self.cache: dict[str, list[float]] = {}
        if cache_path and cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self.api_calls = 0

    def embed_query(self, query: str) -> list[float]:
        key = query.strip()
        if key in self.cache:
            return self.cache[key]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:batchEmbedContents"
        payload = {
            "requests": [
                {
                    "model": f"models/{self.model_name}",
                    "content": {"parts": [{"text": key}]},
                    "taskType": "RETRIEVAL_QUERY",
                }
            ]
        }
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                resp = requests.post(url, params={"key": self.api_key}, json=payload, timeout=30)
                if resp.status_code == 429 and attempt < 3:
                    time.sleep(15)
                    continue
                if not resp.ok:
                    raise RuntimeError(f"Gemini embedding error {resp.status_code}: {resp.text}")
                vec = list(resp.json()["embeddings"][0]["values"])
                self.cache[key] = vec
                self.api_calls += 1
                if self.cache_path:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
                return vec
            except Exception as exc:
                last_err = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Gemini embedding failed: {last_err}")


class Day6Retriever:
    """Unified retriever supporting tfidf, dense, hybrid_rrf, and hybrid_rerank."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
        strategy: str = "hybrid_rrf",
        query_cache_path: Path | None = None,
    ) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.strategy = strategy
        self._build_tfidf()
        self.embedder = GeminiQueryEmbedder(cache_path=query_cache_path) if strategy != "tfidf" else None
        self.rerank_model = "gemini-3.1-flash-lite"
        self.rerank_tokens = 0
        self.rerank_errors = 0

    def _build_tfidf(self) -> None:
        counts = [Counter(tokenize(c["text"])) for c in self.chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(self.chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.tfidf_vecs = [self._tfidf_vec(row) for row in counts]

    def _tfidf_vec(self, counts: Counter[str]) -> dict[str, float]:
        vec = {t: c * self.idf[t] for t, c in counts.items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _search_tfidf(self, query: str, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        q_vec = self._tfidf_vec(Counter(tokenize(query)))
        results = []
        for chunk, vec in zip(self.chunks, self.tfidf_vecs):
            if not matches_filter(chunk, filters):
                continue
            score = sum(val * vec.get(term, 0.0) for term, val in q_vec.items())
            if score > 0:
                results.append({**chunk, "score": round(score, 6), "strategy": "tfidf"})
        return sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]

    def _search_dense(self, query: str, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        assert self.embedder is not None
        q_vec = self.embedder.embed_query(query)
        results = []
        for chunk, c_vec in zip(self.chunks, self.vectors):
            if not matches_filter(chunk, filters):
                continue
            score = cosine_sim(q_vec, c_vec)
            results.append({**chunk, "score": round(score, 6), "strategy": "dense"})
        return sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]

    def _search_hybrid_rrf(self, query: str, filters: dict[str, Any] | None, top_k: int, pool: int = 15, k_rrf: int = 60) -> list[dict[str, Any]]:
        dense_hits = self._search_dense(query, filters, pool)
        tfidf_hits = self._search_tfidf(query, filters, pool)
        rrf_scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            by_id[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_rrf + rank)
        for rank, item in enumerate(tfidf_hits, start=1):
            cid = item["id"]
            by_id.setdefault(cid, item)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_rrf + rank)
        ranked = [
            {**by_id[cid], "score": round(score, 6), "strategy": "hybrid_rrf"}
            for cid, score in sorted(rrf_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        return ranked[:top_k]

    def _rerank_with_llm(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not candidates or len(candidates) <= 1:
            return candidates[:top_k]
        cand_text = "\n\n".join(
            f"[{i}] id={c['id']} (page {c['page']}):\n{c['text'][:600]}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            "Bạn là bộ xếp hạng (reranker) tài liệu. Cho câu hỏi và danh sách các đoạn văn bản được đánh số [0], [1], ...\n"
            "Hãy sắp xếp toàn bộ các chỉ số từ liên quan nhất đến ít liên quan nhất để trả lời câu hỏi.\n"
            "Chỉ trả về JSON hợp lệ dạng: {\"ranked_indices\": [0, 2, 1, ...]}\n\n"
            f"Câu hỏi: {query}\n\nỨng viên:\n{cand_text}"
        )
        for attempt in range(3):
            try:
                text, tokens = gemini_generate(prompt, model=self.rerank_model, json_mode=True, timeout=25)
                self.rerank_tokens += tokens
                data = json.loads(text or "{}")
                indices = [int(i) for i in data.get("ranked_indices", []) if isinstance(i, int) and 0 <= i < len(candidates)]
                seen = set()
                ordered = []
                for idx in indices:
                    if idx not in seen:
                        seen.add(idx)
                        ordered.append({**candidates[idx], "strategy": "hybrid_rerank"})
                for idx, item in enumerate(candidates):
                    if idx not in seen:
                        ordered.append({**item, "strategy": "hybrid_rerank"})
                return ordered[:top_k]
            except Exception:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    self.rerank_errors += 1
        return [{**c, "strategy": "hybrid_rerank"} for c in candidates[:top_k]]

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if self.strategy == "tfidf":
            return self._search_tfidf(query, filters, top_k)
        if self.strategy == "dense":
            return self._search_dense(query, filters, top_k)
        if self.strategy == "hybrid_rrf":
            return self._search_hybrid_rrf(query, filters, top_k)
        if self.strategy == "hybrid_rerank":
            pool = self._search_hybrid_rrf(query, filters, top_k=max(10, top_k))
            return self._rerank_with_llm(query, pool, top_k)
        raise ValueError(f"Unknown strategy: {self.strategy}")

