"""Offline unit tests for Day 7 retrieval and filter logic."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrievers import Day6Retriever, cosine_sim, matches_filter  # noqa: E402


class _FakeEmbedder:
    def __init__(self, mapping):
        self.mapping = mapping

    def embed_query(self, query):
        return self.mapping[query]


class Day7RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": "c1",
                "text": "Graham khuyên nhà đầu tư phòng vệ giữ tỷ lệ 25% đến 75% cổ phiếu.",
                "source_id": "book-1",
                "source": "book.pdf",
                "page": 14,
                "metadata": {"author_role": "graham", "section": "chapter_1"},
            },
            {
                "id": "c2",
                "text": "Jason Zweig bình luận về Isaac Newton mất 20000 bảng ở South Sea.",
                "source_id": "book-1",
                "source": "book.pdf",
                "page": 19,
                "metadata": {"author_role": "zweig", "section": "commentary_intro"},
            },
            {
                "id": "c3",
                "text": "Warren Buffett đọc cuốn sách lần đầu năm 1950 khi mười chín tuổi.",
                "source_id": "book-1",
                "source": "book.pdf",
                "page": 33,
                "metadata": {"author_role": "buffett", "section": "preface_buffett"},
            },
        ]
        self.vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    def test_matches_filter(self):
        self.assertTrue(matches_filter(self.chunks[0], None))
        self.assertTrue(matches_filter(self.chunks[0], {"author_role": "graham"}))
        self.assertFalse(matches_filter(self.chunks[0], {"author_role": "zweig"}))
        self.assertTrue(matches_filter(self.chunks[2], {"page": 33}))

    def test_cosine_sim(self):
        self.assertAlmostEqual(cosine_sim([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_sim([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_tfidf_and_hybrid_rrf_with_filter(self):
        r = Day6Retriever(self.chunks, self.vectors, strategy="tfidf")
        hits = r.search("Isaac Newton South Sea", filters={"author_role": "zweig"}, top_k=3)
        self.assertEqual([h["id"] for h in hits], ["c2"])

        r_hybrid = Day6Retriever(self.chunks, self.vectors, strategy="tfidf")
        r_hybrid.strategy = "hybrid_rrf"
        r_hybrid.embedder = _FakeEmbedder({"Warren Buffett": [0.0, 0.0, 1.0]})
        rrf_hits = r_hybrid.search("Warren Buffett", filters=None, top_k=2)
        self.assertEqual(rrf_hits[0]["id"], "c3")
        self.assertEqual(rrf_hits[0]["strategy"], "hybrid_rrf")


if __name__ == "__main__":
    unittest.main()

