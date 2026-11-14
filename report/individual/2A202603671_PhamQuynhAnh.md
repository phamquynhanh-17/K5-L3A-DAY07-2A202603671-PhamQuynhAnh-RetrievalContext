# Báo cáo Day 7 — Agentic RAG

- **Họ và tên:** Phạm Quỳnh Anh
- **MSSV:** `2A202603671`
- **Lớp:** `L3A`

---

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Kế thừa trực tiếp dữ liệu từ Day 6: trích đoạn bản dịch tiếng Việt cuốn *Nhà đầu tư thông minh (The Intelligent Investor)* gồm Lời nói đầu của Warren E. Buffett, tưởng niệm Benjamin Graham, Phần mở đầu và Chương 1 cùng chú giải của Jason Zweig (`sách.pdf`, dùng nội bộ cho mục đích học tập; không commit bản PDF gốc có bản quyền lên repo công khai).
  - File có `33` trang vật lý, trong đó **`32` trang nội dung hợp lệ** ($\ge 100$ ký tự; bỏ 1 trang chú thích tái bản 65 ký tự).
  - Sử dụng **Chiến lược C** từ Day 6: `by_paragraph(max_size=800)` kèm metadata (`document_id`, `source`, `page`, `book_page`, `section`, `author_role`), gồm **`171` chunk** (độ dài tối thiểu `113`, trung vị `585`, tối đa `797` ký tự).
- **Vector store, embedding model, phiên bản và số chiều:**
  - Vector store Day 6: `chromadb==1.5.9` (`chroma_db/chroma.sqlite3`, collection `day06_c`, metric `cosine`), kết hợp vector cache `embeddings_cache.json` (`171` vector). Script [src/prepare_day6.py](../../src/prepare_day6.py) trích xuất đồng bộ chunk + vector sang `starter/data/day6_c/`.
  - Embedding model: **`gemini-embedding-001`** (Google Gemini API, `3072` chiều), dùng `taskType="RETRIEVAL_DOCUMENT"` khi index ở Day 6 và `taskType="RETRIEVAL_QUERY"` khi truy vấn ở Day 7.
- **Agent model, cách chạy và phân chia dev/test:**
  - Agent controller & reranker: **`gemini-3.1-flash-lite`** gọi qua Gemini REST API (`temperature=0.0`, `responseMimeType="application/json"`, `timeout=25s`, có retry khi gặp `429`/`5xx`).
  - Khung điều phối: [starter/agentic.py](../../starter/agentic.py) (`run_agent`) kết nối với `GeminiController` và `Day6Retriever` trong [src/agent_runner.py](../../src/agent_runner.py). Chi tiết các lệnh tái lập xem [src/README.md](../../src/README.md).
  - Phân chia benchmark trước khi thử nghiệm (`24` câu trong `starter/data/day6_c/agent_cases.json`):
    - **Dev (`17` câu):** `15` câu có đáp án (`q01`–`q15`, trong đó `q04`, `q09`, `q14` có metadata filter) + `2` câu thiếu bằng chứng (`u01`, `u02`).
    - **Test (`7` câu):** `5` câu có đáp án (`q16`–`q20`, trong đó `q18` có metadata filter) + `2` câu thiếu bằng chứng (`u03`, `u04`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Toàn bộ `20` câu có đáp án kế thừa từ Day 6, mỗi câu có `expected_answer`, `gold` (`source_id`, `page`, `evidence`) nằm trọn trong đúng 1 chunk và được kiểm tra tự động qua `validate()` trong [starter/baseline.py](../../starter/baseline.py). Tuyệt đối không đưa `gold` hay `expected_answer` vào agent hoặc retriever.
  - `4` câu thiếu bằng chứng (`u01`–`u04`) chọn gần chủ đề sách đầu tư (Chương 8 *Ngài Thị trường*, Chương 20 *giá trị nội tại cổ phiếu tăng trưởng*, Phụ lục *năm 2008*, Chương 14 *7 tiêu chí định lượng*). Đã kiểm tra toàn bộ `171` chunk bằng tìm kiếm từ khóa (`ngài thị trường`, `mr. market`, `giá trị nội tại`, `cổ phiếu tăng trưởng`, `phụ lục`, `2008`, `chương 14`, `tiêu chí định lượng`) — tất cả đều trả về `0` chunk.
  - Chấm thủ công câu trả lời (`0`/`1`/`2`), độ hỗ trợ của từng citation (`True`/`False` bằng cách đối chiếu trực tiếp văn bản chunk được trích dẫn) và xác nhận abstention (`True`/`False`) rồi tổng hợp bằng [starter/agent_metrics.py](../../starter/agent_metrics.py).

---

## Retrieval và thử nghiệm

Đã triển khai và so sánh **4 phương pháp retrieval** trên cùng tập `171` chunk Strategy C ([src/retrievers.py](../../src/retrievers.py), [src/benchmark_retrieval.py](../../src/benchmark_retrieval.py)):
1. **TF-IDF (`tfidf`)** — baseline lexical theo [starter/baseline.py](../../starter/baseline.py).
2. **Dense (`dense`)** — cosine similarity trên vector 3072 chiều Day 6 (`gemini-embedding-001`), có áp dụng metadata filter.
3. **Hybrid RRF (`hybrid_rrf`)** — hợp nhất top-15 Dense và top-15 TF-IDF bằng Reciprocal Rank Fusion ($k=60$), có metadata filter.
4. **Hybrid + LLM Rerank (`hybrid_rerank`)** — lấy top-10 ứng viên từ `hybrid_rrf`, dùng `gemini-3.1-flash-lite` xếp hạng lại top-3 theo ngữ nghĩa trực tiếp của câu hỏi.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | `tfidf`, top-k=3, no filter, no rerank, budget=800 ký tự | Mốc đối chiếu lexical ban đầu | dev | 10/15 (66.7%) | 13/15 (86.7%) | 8/15 (53.3%) | median 0.71 ms / \$0 |
| Phương pháp 2 | `dense` (Day 6 3072-dim), top-k=3, filter, no rerank, budget=800 | Embedding ngữ nghĩa xử lý câu diễn đạt lại tốt hơn TF-IDF | dev | 11/15 (73.3%) | 14/15 (93.3%) | 9/15 (60.0%) | median 670.59 ms / \$0 (Free Tier) |
| Phương pháp 3 | `hybrid_rrf` (Dense + TF-IDF, RRF $k=60$), top-k=3, filter, budget=800 | Kết hợp khớp từ khóa chính xác và ngữ nghĩa giúp tăng Hit@1 | dev | 12/15 (80.0%) | 14/15 (93.3%) | 9/15 (60.0%) | median 118.42 ms / \$0 (cached emb) |
| Cải thiện 1 | `hybrid_rrf`, top-k=3, filter, **budget=2400 ký tự** | Khắc phục lỗi `budget=800` chỉ chứa được ~1 chunk làm rớt gold rank 2–3 | dev | 12/15 (80.0%) | 14/15 (93.3%) | **14/15 (93.3%)** | median 120.11 ms / \$0 |
| Cải thiện 2 (Phương pháp 4) | `hybrid_rerank` (RRF top-10 → `gemini-3.1-flash-lite` rerank top-3), filter, **budget=2400** | LLM đọc trực tiếp ứng viên để xử lý ca lệch rank ở câu hỏi tác giả/chương (`q02`) | dev | **15/15 (100.0%)** | **15/15 (100.0%)** | **15/15 (100.0%)** | median 1292.71 ms / 28,657 tokens (\$0 Free Tier) |
| **Cấu hình đã chọn** | `hybrid_rerank`, top-k=3, filter, **budget=3600** (đã chốt trên dev) | Đánh giá độc lập trên test sau khi chốt | **test** | **5/5 (100.0%)** | **5/5 (100.0%)** | **5/5 (100.0%)** | median 1867.69 ms / 9,693 tokens (\$0) |
| Baseline | `tfidf`, top-k=3, no filter, budget=800 | Đối chiếu độc lập trên test | **test** | 5/5 (100.0%) | 5/5 (100.0%) | 4/5 (80.0%) | median 0.60 ms / \$0 |

### Chấm chất lượng Context trên 5 câu Test có đáp án (thang 0/1/2)
- **Baseline (`test-tfidf`, budget=800):**
  - `q16` (`2`): có `p33_para_0` đủ nguồn/điều kiện.
  - `q17` (`2`): có `p29_para_0` đủ thông tin 250% và lừa đảo tháng 10.
  - `q18` (`2`): có `p1_para_6` đủ thông tin chi phí môi giới 8%.
  - `q19` (`2`): có `p4_para_3` đủ số liệu 1973 và 2002.
  - `q20` (`0`): dù `p14_para_1` đứng rank #1, `pack_context(..., budget=800)` loại bỏ chunk này vì dài gần 800 ký tự (cộng header vượt budget) và chỉ giữ rank #2 (`p1_para_3`), nên context không có evidence.
  - *Tổng hợp Baseline test context:* Mức `0`: **1**, Mức `1`: **0**, Mức `2`: **4** (TB `1.60/2`).
- **Cấu hình đã chọn (`test-hybrid-rerank-b3600`):**
  - Cả 5/5 câu (`q16`–`q20`) đều đạt mức **`2`**: `p33_para_0`, `p29_para_0`, `p1_para_6`, `p4_para_3`, `p14_para_1` đều đứng ở rank #1 và nằm trọn trong context kèm đủ chunk lân cận/nguồn.
  - *Tổng hợp Chosen test context:* Mức `0`: **0**, Mức `1`: **0**, Mức `2`: **5** (TB `2.00/2`).

### Phân tích 2 lỗi dev và lý do chọn cấu hình
1. **Lỗi dev #1 — Tràn context budget làm rớt luôn chunk hạng #1 (`q10`, `q14` ở `dev-tfidf` và `q08`, `q10` ở `dev-dense-b800`):**
   - Trong `starter/results/dev-tfidf/traces.json`, `q10` có `hit1=True` (`p17_para_4` đứng hạng #1) nhưng `context_ids` lại chỉ chứa `p5_para_2` (hạng #3). Nguyên nhân: các chunk Strategy C có trung vị `585` ký tự (tối đa `797` ký tự), khi cộng thêm tiêu đề `[chunk_id | source | page]` bị vượt `budget=800`, khiến `pack_context` bỏ qua luôn rank #1 và #2.
   - **Thử nghiệm & kết quả:** Ở Cải thiện 1 (`dev-hybrid-b2400`), tăng `budget` từ `800` lên `2400` ký tự (đủ chứa cả 3 chunk top-3, trung bình `1919.2` ký tự) giúp `Context Evidence Rate` trên `dev` tăng từ **9/15 (60.0%)** lên **14/15 (93.3%)**.
2. **Lỗi dev #2 — Lệch thứ hạng ngữ nghĩa ở câu hỏi nhắc tên tác giả/chương (`q02` trong `dev-dense-b800` và `dev-hybrid-b2400`):**
   - `q02` hỏi hai chương Buffett khuyên đặc biệt chú ý (`p33_para_1`), nhưng cả `dense` lẫn `hybrid_rrf` đều xếp `p33_para_0` (đoạn mở đầu có tên Warren E. Buffett) lên #1 và đẩy `p33_para_1` ra ngoài top-3 (chỉ nằm trong top-10 ứng viên RRF).
   - **Thử nghiệm & kết quả:** Ở Cải thiện 2 (`dev-hybrid-rerank-b2400`), dùng `gemini-3.1-flash-lite` xếp hạng lại top-10 ứng viên RRF đưa `p33_para_1` lên đúng hạng #1, nâng `Hit@1`, `Hit@3` và `Context Evidence Rate` trên `dev` lên **15/15 (100.0%)**.
3. **Lý do chốt cấu hình (`hybrid_rerank`, `top_k=3`, `budget=3600`, `max_searches=4`):**
   - Trên `dev`, `hybrid_rerank` đạt độ chính xác retrieval cao nhất (`15/15` mọi chỉ số). Khi ghép vào agent đa lượt tìm, nâng thêm `budget=3600` (so với `2400`) để tránh tình trạng chunk lượt 1 chiếm hết budget khiến lượt tìm thứ 2 không nạp thêm được đoạn mới (đặc biệt ở các câu thiếu bằng chứng cần tìm nhiều lần như `u04`).

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - **Tool:** Agent gọi `Day6Retriever.search(query, filters, top_k=3)` thông qua action `{"action": "search", "query": "..."}`.
  - **Điều kiện dừng:** Agent trả về `{"action": "answer", "answer": "... [chunk_id]", "citations": [...]}` khi đã đủ bằng chứng, hoặc từ chối khi thiếu bằng chứng (`citations: []`). `run_agent` cũng tự động chuyển sang `abstain` nếu vượt quá `max_searches` (`2` cho baseline, `4` cho cấu hình chọn), context rỗng, hoặc citation không nằm trong context/không xuất hiện dạng `[chunk_id]` trong câu trả lời.
  - **Cách chọn context:** `pack_context` khử trùng lặp theo `chunk_id` và giữ các đoạn trong phạm vi `budget` (`800` ở baseline; `2400` ở Cải thiện 1; `3600` ở Cải thiện 2 / Chosen).
  - **Xử lý lỗi:** Lỗi kết nối/API không bị che giấu hay thay bằng kết quả giả (`status: "error"` được báo riêng trong `provider_errors`, không tính vào mẫu số chất lượng). Ở Cải thiện 2, đã bổ sung cơ chế chờ `25s` khi gặp HTTP `429` và `timeout=25s` cho từng request.

| Cấu hình / split | Tool use | Số lượt gọi (phân bố) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline (`tfidf`, b=800, max=2) / **dev** | 17/17 (100%, 0 lỗi) | `{1: 17}` | 0: 6, 1: 3, 2: 6 — TB: **1.00 / 2** | 9/9 (100%) | 9/9 (100%) | 2/2 (100%) | 6/15 (40.0%) |
| Cải thiện 1 (`hybrid_rrf`, b=2400, max=4) / **dev** | 16/16 (100%, **1 lỗi 429**) | `{1: 16}` | 0: 0, 1: 1, 2: 13 — TB: **1.93 / 2** | 17/17 (100%) | 15/15 (100%) | 1/2 (50.0%) | 0/14 (0%) |
| Cải thiện 2 (`hybrid_rerank`, b=3600, max=4) / **dev** | 17/17 (100%, 0 lỗi) | `{1: 17}` | 0: 0, 1: 1, 2: 14 — TB: **1.93 / 2** | 17/17 (100%) | 15/15 (100%) | **2/2 (100%)** | **0/15 (0%)** |
| Baseline (`tfidf`, b=800, max=2) / **test** | 7/7 (100%, 0 lỗi) | `{1: 6, 2: 1}` | 0: 1, 1: 0, 2: 4 — TB: **1.60 / 2** | 4/4 (100%) | 4/4 (100%) | 2/2 (100%) | 1/5 (20.0%) |
| **Cấu hình đã chọn** (`hybrid_rerank`, b=3600, max=4) / **test** | 7/7 (100%, 0 lỗi) | `{1: 6, 3: 1}` | 0: 0, 1: 0, 2: 5 — TB: **2.00 / 2** | **5/5 (100%)** | **5/5 (100%)** | **2/2 (100%)** | **0/5 (0%)** |

### Bảng kiểm tra từng câu Test (7/7 câu — Cấu hình đã chọn `test-agent-chosen`)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án | 1 lượt `hybrid_rerank` → `[p33_para_0, p20_para_3, p19_para_1]` | Không đòi hỏi IQ cao chót vót, tầm hiểu biết sâu sắc khác thường hay thông tin nội bộ `[nha-dau-tu-thong-minh-excerpt_p33_para_0]` | **Answer: 2, Cite: True.** Đúng trực tiếp câu hỏi Có/Không theo Lời nói đầu Buffett (tr.33); chunk trích dẫn khớp nguyên văn. |
| `q17` | Có đáp án | 1 lượt `hybrid_rerank` → `[p29_para_0, p29_para_2, p29_para_1]` | Lãi 250% ngày đầu với Savold Tire (4/1919), tới tháng 10 bị phát hiện lừa đảo và cổ phiếu vô giá trị `[nha-dau-tu-thong-minh-excerpt_p29_para_0]` | **Answer: 2, Cite: True.** Đủ cả 2 vế câu hỏi, `p29_para_0` hỗ trợ chính xác từng chi tiết. |
| `q18` | Có đáp án (filter) | 1 lượt `hybrid_rerank` (`section=commentary_ch1`) → `[p1_para_6, p1_para_5, p1_para_2]` | Do nhiều người biết mua vào tháng 12 và chi phí môi giới chứng khoán nhỏ ăn tới 8% khoản đầu tư `[nha-dau-tu-thong-minh-excerpt_p1_para_6]` | **Answer: 2, Cite: True.** Trả lời đầy đủ lý do theo `p1_para_6`, tuân thủ metadata filter. |
| `q19` | Có đáp án | 1 lượt `hybrid_rerank` → `[p4_para_3, p17_para_3, p21_para_3]` | Năm 1973 mức luân chuyển là 20%, đến năm 2002 tăng lên 105% `[nha-dau-tu-thong-minh-excerpt_p4_para_3]` | **Answer: 2, Cite: True.** Đúng cả hai mốc số liệu trên NYSE, `p4_para_3` hỗ trợ trực tiếp. |
| `q20` | Có đáp án | 1 lượt `hybrid_rerank` → `[p16_para_0, p14_para_1, p23_para_2]` | Dành khoản vốn càng nhỏ càng tốt trong quỹ riêng, tuyệt đối không đổ thêm tiền khi thị trường lên `[nha-dau-tu-thong-minh-excerpt_p14_para_1]` | **Answer: 2, Cite: True.** Khắc phục hoàn toàn lỗi từ chối nhầm ở Baseline test (`budget=800`). |
| `u03` | Thiếu bằng chứng | 1 lượt `hybrid_rerank` → `[p12_para_2, p14_para_4, p25_para_5]` | `"Không đủ bằng chứng để trả lời."` (`citations: []`) | **Abstain: True.** Từ chối chính xác vì tài liệu không có Phụ lục và không có dữ liệu 2008. |
| `u04` | Thiếu bằng chứng | **3 lượt** `hybrid_rerank` đổi truy vấn (Chương 14 / Benjamin Graham) → 4 chunk trong budget 3600 | `"Không đủ bằng chứng để trả lời."` (`citations: []`) | **Abstain: True.** Agent tự thử lại 3 lượt để kiểm tra kỹ rồi mới từ chối chính xác. |

### Đối chiếu 7 câu Test trên Baseline (`test-agent-baseline`)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án | 1 lượt `tfidf` → context `['p33_para_0']` | Không đòi hỏi IQ cao chót vót, tầm hiểu biết sâu sắc hay thông tin nội bộ `[nha-dau-tu-thong-minh-excerpt_p33_para_0]` | **Answer: 2, Cite: True.** Đúng trực tiếp câu hỏi và nguồn hỗ trợ. |
| `q17` | Có đáp án | 1 lượt `tfidf` → context `['p29_para_0']` | Lãi 250% ngày đầu (4/1919), tới tháng 10 bị phát hiện lừa đảo và vô giá trị `[nha-dau-tu-thong-minh-excerpt_p29_para_0]` | **Answer: 2, Cite: True.** Đủ hai vế, nguồn hỗ trợ chính xác. |
| `q18` | Có đáp án (filter) | 1 lượt `tfidf` → context `['p1_para_6']` | Mua trước vào tháng 12 và phí môi giới chứng khoán nhỏ ăn tới 8% `[nha-dau-tu-thong-minh-excerpt_p1_para_6]` | **Answer: 2, Cite: True.** Đủ lý do và đúng nguồn. |
| `q19` | Có đáp án | 1 lượt `tfidf` → context `['p4_para_3']` | Năm 1973 là 20%, năm 2002 tăng lên 105% `[nha-dau-tu-thong-minh-excerpt_p4_para_3]` | **Answer: 2, Cite: True.** Đúng số liệu 2 năm trên NYSE. |
| `q20` | Có đáp án | 1 lượt `tfidf` → `p14_para_1` (rank #1) bị loại do vượt `budget=800`, chỉ còn `['p1_para_3']` | `"Không đủ bằng chứng để trả lời."` (`citations: []`) | **Answer: 0 (Từ chối nhầm).** Context mất `p14_para_1` nên agent buộc phải từ chối. |
| `u03` | Thiếu bằng chứng | 1 lượt `tfidf` → context `['p12_para_2']` | `"Không đủ bằng chứng để trả lời."` (`citations: []`) | **Abstain: True.** Từ chối đúng câu Phụ lục / 2008. |
| `u04` | Thiếu bằng chứng | **2 lượt** `tfidf` → context `['p9_para_1']` | `"Không đủ bằng chứng để trả lời."` (`citations: []`) | **Abstain: True.** Thử 2 lượt đến giới hạn `max_searches=2` rồi từ chối đúng. |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Dữ liệu & benchmark: `starter/data/day6_c/` (`chunks.json`, `vectors.json`, `questions.json`, `agent_cases.json`).
  - Trace & audit từng cấu hình nằm tại `starter/results/`: `dev-agent-baseline`, `dev-agent-hybrid-b2400`, `dev-agent-hybrid-rerank-b3600`, `test-agent-baseline`, `test-agent-chosen` (mỗi thư mục gồm `traces.json`, `reviews.json`, `summary.json`, `run_meta.json`).
  - Thời gian & token:
    - `dev-agent-baseline`: `84.65s` tổng (`median 2,236 ms/câu`), `18,375 tokens`, `$0`.
    - `dev-agent-hybrid-b2400`: `138.56s` tổng (`median 3,194 ms/câu`), `24,962 tokens`, `1` lỗi `429`, `$0`.
    - `dev-agent-hybrid-rerank-b3600`: `225.71s` tổng (`median 5,067 ms/câu`), `60,784 tokens`, `0` lỗi, `$0`.
    - `test-agent-baseline`: `42.48s` tổng (`median 2,344 ms/câu`), `9,160 tokens`, `0` lỗi, `$0`.
    - `test-agent-chosen`: `63.72s` tổng (`median 4,685 ms/câu`), `31,468 tokens`, `0` lỗi, `$0`.
  - Lệnh chạy tái lập chi tiết nằm tại [src/README.md](../../src/README.md).

- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - **Trước → sau trên `dev`:** Baseline chỉ đạt Grounded Answer TB `1.00/2` vì từ chối nhầm `6/15` câu (`q02, q04, q06, q08, q10, q11`) và thiếu ý ở `q13, q14` do `budget=800`. Chuyển sang `hybrid_rerank` + `budget=3600` đưa Grounded Answer TB lên `1.93/2` (`14` câu điểm 2, `1` câu điểm 1), giảm từ chối nhầm xuống `0/15`.
  - **Ví dụ lỗi thực tế phát hiện trên `dev` ở Cải thiện 1 (`dev-agent-hybrid-b2400`) và cách sửa:**
    1. *Lỗi API `429` (`q01`):* Gọi liên tục không giãn cách và retry chỉ 2–4s làm `q01` văng lỗi quota. Sau khi thêm retry `25s` trên mã `429` và nghỉ `0.5s` giữa các câu, Cải thiện 2 và Test đạt **`0` lỗi**.
    2. *Lỗi gán citation khi từ chối (`u01`):* Trong Cải thiện 1, model trả lời đúng về ngữ nghĩa là `"Không đủ bằng chứng trong tài liệu để giải thích chi tiết khái niệm 'Ngài Thị trường'..."` nhưng vẫn đính kèm `[p33_para_1, p17_para_4]` (vì các đoạn này có nhắc thoáng qua "chương 8"). Hệ quả là `run_agent` ghi nhận `status="answered"`, làm `Abstention đúng` tụt xuống `1/2`. Sau khi siết rule prompt ở Cải thiện 2 (cấm đính citation khi từ chối nội dung trọng tâm chưa xuất hiện), `u01` và mọi câu thiếu bằng chứng ở `dev` lẫn `test` đều đạt `status="abstain"` (`2/2`).
    3. *Lưu ý câu `q03` trên `dev` (điểm `1/2`):* Câu hỏi viết *"Ngày nào trong cuộc đời Ben Graham..."* nên agent trả lời rất trung thực rằng tài liệu không ghi ngày cụ thể mà chỉ nói khi ông gần 80 tuổi (`p32_para_0`), bỏ sót 3 việc muốn làm mỗi ngày trong `expected_answer`. Điều này phản ánh sự nhạy cảm của agent với câu chữ câu hỏi thay vì lỗi retrieval.
  - **Đánh đổi & giới hạn:**
    - `hybrid_rerank` + `budget=3600` đạt Grounded Answer `2.00/2` trên `test` nhưng đánh đổi độ trễ gấp ~2 lần (`median 4.68s` so với `2.34s`) và tiêu tốn gấp ~3.4 lần token (`31,468` so với `9,160`) so với TF-IDF baseline.
    - Ngoài ra, dữ liệu `day06_c` là trích đoạn 32 trang tiếng Việt có một số lỗi dấu OCR có sẵn trong PDF gốc (ví dụ `"đău tư"`, `"thẩp"`); ở các corpus hàng nghìn trang, cần nâng cấp `hybrid_rerank` theo batch hoặc dùng reranker chuyên biệt (Cross-Encoder) để giảm chi phí token LLM.

---

## Thử thách nâng cao (không thực hiện)

Bài làm thực hiện **Phần cơ bản (85 điểm)** với dữ liệu 32 trang từ Day 6 theo định hướng đã chọn; không thực hiện thử thách nâng cao 100 trang.
