"""Extract Day 6 Strategy C chunks + embeddings from Chroma SQLite & cache, and convert Day 6 questions."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAY6 = ROOT / "data" / "Lab_06" / "K5-L3-Day06-DataIngestionEngineering"
CHROMA_DB = DAY6 / "starter" / "results" / "chroma_db" / "chroma.sqlite3"
EMB_CACHE = DAY6 / "starter" / "results" / "embeddings_cache.json"
DAY6_QUESTIONS = DAY6 / "benchmark" / "questions.json"
OUT_DIR = ROOT / "starter" / "data" / "day6_c"


def extract_chunks_from_chroma(db_path: Path, collection_name: str = "day06_c") -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # Get all embeddings in collection
    rows = cur.execute(
        """
        SELECT e.id, e.embedding_id
        FROM embeddings e
        JOIN segments s ON e.segment_id = s.id
        JOIN collections c ON s.collection = c.id
        WHERE c.name = ?
        ORDER BY e.id
        """,
        (collection_name,),
    ).fetchall()

    chunks = []
    for row_id, chunk_id in rows:
        meta_rows = cur.execute(
            "SELECT key, string_value, int_value, float_value, bool_value FROM embedding_metadata WHERE id = ?",
            (row_id,),
        ).fetchall()
        text = ""
        meta = {}
        for key, s_val, i_val, f_val, b_val in meta_rows:
            val = s_val if s_val is not None else (i_val if i_val is not None else (f_val if f_val is not None else b_val))
            if key == "chroma:document":
                text = str(val or "")
            else:
                meta[key] = val

        source_id = str(meta.pop("document_id", "nha-dau-tu-thong-minh-excerpt"))
        source = str(meta.pop("source", "data/sách.pdf"))
        page = int(meta.pop("page", 0))
        chunks.append(
            {
                "id": chunk_id,
                "text": text,
                "source_id": source_id,
                "source": source,
                "page": page,
                "metadata": meta,
            }
        )
    conn.close()
    return chunks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = extract_chunks_from_chroma(CHROMA_DB, "day06_c")
    emb_data = json.loads(EMB_CACHE.read_text(encoding="utf-8"))
    vectors = emb_data["vectors"]["B"]
    if len(chunks) != len(vectors):
        raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors in cache B")

    # Save chunks.json and vectors.json
    (OUT_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "vectors.json").write_text(
        json.dumps({"model": emb_data["model"], "dimensions": len(vectors[0]), "vectors": vectors}),
        encoding="utf-8",
    )

    # Convert Day 6 questions
    raw_q = json.loads(DAY6_QUESTIONS.read_text(encoding="utf-8"))
    questions = []
    for q in raw_q:
        item = {
            "id": q["id"],
            "split": q["split"],
            "type": q.get("type", "unknown"),
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": [
                {
                    "source_id": q["gold_source_id"],
                    "page": q["gold_page"],
                    "evidence": q["evidence_quote"],
                }
            ],
        }
        if q.get("filter"):
            item["filter"] = q["filter"]
        questions.append(item)

    (OUT_DIR / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build agent_cases.json (20 answerable + 4 unanswerable = 24 total: 17 dev, 7 test)
    agent_cases = [
        {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "expected_answer": q["expected_answer"],
            "gold": q["gold"],
            **({"filter": q["filter"]} if "filter" in q else {}),
        }
        for q in questions
    ]
    unanswerable = [
        {
            "id": "u01",
            "split": "dev",
            "question": "Trong Chương 8 của cuốn sách, Benjamin Graham giải thích khái niệm 'Ngài Thị trường' (Mr. Market) như thế nào?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Trích đoạn 32 trang chỉ gồm Lời nói đầu và Chương 1 (cùng bình luận); Chương 8 chỉ được Buffett nhắc tên ở trang 33, không có nội dung 'Ngài Thị trường'.",
        },
        {
            "id": "u02",
            "split": "dev",
            "question": "Trong Chương 20, công thức tính giá trị nội tại của cổ phiếu tăng trưởng mà Graham đưa ra là gì?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Chương 20 chỉ được nhắc tên ở trang 33; từ khóa 'giá trị nội tại' và 'cổ phiếu tăng trưởng' không xuất hiện trong 171 chunk.",
        },
        {
            "id": "u03",
            "split": "test",
            "question": "Trong Phụ lục của sách, tỷ lệ phân bổ trái phiếu kho bạc Mỹ kỳ hạn 10 năm năm 2008 là bao nhiêu?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu không có Phụ lục và không có dữ liệu năm 2008 (kiểm tra từ khóa 'phụ lục' và '2008' đều ra 0 chunk).",
        },
        {
            "id": "u04",
            "split": "test",
            "question": "Theo Chương 14 của Graham, 7 tiêu chí định lượng để chọn cổ phiếu cho nhà đầu tư phòng vệ gồm những gì?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Trích đoạn chỉ đến hết Chương 1; từ khóa 'chương 14' và 'tiêu chí định lượng' đều ra 0 chunk.",
        },
    ]
    agent_cases.extend(unanswerable)
    (OUT_DIR / "agent_cases.json").write_text(json.dumps(agent_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Extracted {len(chunks)} chunks ({len(vectors[0])} dims), {len(questions)} retrieval questions, and {len(agent_cases)} agent cases to {OUT_DIR}")


if __name__ == "__main__":
    main()
