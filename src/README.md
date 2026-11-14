# Code Day 7 — Agentic RAG

Các module chính trong `src/`:

| File | Công dụng |
|---|---|
| `prepare_day6.py` | Trích xuất 171 chunk Strategy C + vector 3072 chiều từ ChromaDB & cache Day 6; tạo `questions.json` (15 dev / 5 test) và `agent_cases.json` (24 câu: 17 dev / 7 test) tại `starter/data/day6_c/`. |
| `check_gemini.py` | Đọc `GEMINI_API_KEY` từ `.env` và cung cấp hàm gọi Gemini REST API (`gemini-3.1-flash-lite`) kèm retry 429/5xx và timeout rõ ràng. |
| `retrievers.py` | Cài đặt 4 phương pháp retrieval qua lớp `Day6Retriever`: `tfidf`, `dense` (trên vector store Day 6 3072-dim `gemini-embedding-001`), `hybrid_rrf` (Dense + TF-IDF RRF k=60), và `hybrid_rerank` (RRF top-10 + LLM rerank). |
| `benchmark_retrieval.py` | Chạy benchmark retrieval trên `dev` hoặc `test`, tính `Hit@1`, `Evidence Hit@3`, `Context Evidence Rate`, độ dài context và thời gian/token. |
| `agent_runner.py` | Hiện thực `GeminiController` (real LLM `gemini-3.1-flash-lite`) nối với `starter/agentic.py` (`run_agent`) và `Day6Retriever`. |
| `inspect_traces.py` | In câu hỏi, đáp án kỳ vọng, câu trả lời và đoạn chunk được trích dẫn để kiểm tra thủ công. |
| `write_reviews_and_aggregate.py` | Lưu nhãn audit thủ công (`reviews.json`) và chạy `starter/agent_metrics.py` để tổng hợp chỉ số agent. |
| `demo.py` | Chạy demo trực tiếp 3 tình huống theo `GUIDE.md` (1 câu có bằng chứng, 1 câu thiếu bằng chứng, 1 ca lỗi & cải thiện) hoặc câu hỏi tùy chọn `--question "..."`. |
| `chatbot_app.py` | Khởi chạy Web Chatbot UI tại `http://127.0.0.1:8000` để hỏi-đáp trực tiếp trên trình duyệt kèm trace và trích dẫn. |

## Lệnh chạy tái lập

Chạy từ thư mục gốc của repo (`K5-L3-Day07-Retrieval-Context-Engineering`):

```bash
# 0. Cài thư viện và cấu hình API key (.env từ .env.example)
pip install -r requirements.txt
cp .env.example .env  # điền GEMINI_API_KEY của bạn

# 1. Chạy unit test (offline)
python -m unittest discover -s tests -v

# 2. (Tùy chọn nếu có sẵn thư mục Day 6) Tái tạo dữ liệu -> starter/data/day6_c/
# Repo đã kèm sẵn starter/data/day6_c/{chunks,vectors,questions,agent_cases}.json để chạy ngay
python src/prepare_day6.py

# 3. Kiểm tra kết nối Gemini API (cần GEMINI_API_KEY trong .env)
python src/check_gemini.py

# 4. Chạy Retrieval benchmarks trên dev
python starter/baseline.py --chunks starter/data/day6_c/chunks.json --questions starter/data/day6_c/questions.json --split dev --output starter/results/dev-tfidf
python src/benchmark_retrieval.py --strategy dense --split dev --budget 800 --output starter/results/dev-dense-b800
python src/benchmark_retrieval.py --strategy hybrid_rrf --split dev --budget 800 --output starter/results/dev-hybrid-b800
python src/benchmark_retrieval.py --strategy hybrid_rrf --split dev --budget 2400 --output starter/results/dev-hybrid-b2400
python src/benchmark_retrieval.py --strategy hybrid_rerank --split dev --budget 2400 --output starter/results/dev-hybrid-rerank-b2400

# 5. Chạy Agent trên dev (baseline và các cấu hình cải thiện)
python src/agent_runner.py --strategy tfidf --split dev --top-k 3 --budget 800 --max-searches 2 --output starter/results/dev-agent-baseline
python src/agent_runner.py --strategy hybrid_rrf --split dev --top-k 3 --budget 2400 --max-searches 4 --output starter/results/dev-agent-hybrid-b2400
python src/agent_runner.py --strategy hybrid_rerank --split dev --top-k 3 --budget 3600 --max-searches 4 --output starter/results/dev-agent-hybrid-rerank-b3600

# 6. Sau khi chốt cấu hình trên dev, chạy Baseline và Cấu hình chọn trên test
python starter/baseline.py --chunks starter/data/day6_c/chunks.json --questions starter/data/day6_c/questions.json --split test --output starter/results/test-tfidf
python src/benchmark_retrieval.py --strategy hybrid_rerank --split test --top-k 3 --budget 3600 --output starter/results/test-hybrid-rerank-b3600
python src/agent_runner.py --strategy tfidf --split test --top-k 3 --budget 800 --max-searches 2 --output starter/results/test-agent-baseline
python src/agent_runner.py --strategy hybrid_rerank --split test --top-k 3 --budget 3600 --max-searches 4 --output starter/results/test-agent-chosen

# 7. Tổng hợp nhãn audit thủ công
python src/write_reviews_and_aggregate.py

# 8. Chạy demo CLI hoặc Web Chatbot UI
python src/demo.py
python src/chatbot_app.py --host 127.0.0.1 --port 8000
```
