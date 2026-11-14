# Nhãn kiểm tra thủ công (Manual Audit) — Day 7

Tài liệu này ghi lại kết quả đọc đối chiếu trực tiếp:
1. Chấm **Context (0/1/2)** trên 5 câu `test` có đáp án (`q16`–`q20`) giữa Baseline (`test-tfidf`) và Cấu hình đã chọn (`test-hybrid-rerank-b3600`).
2. Chấm **Agent & Citation** trên toàn bộ 7 câu `test` (`q16`–`q20`, `u03`–`u04`) và 17 câu `dev` (`q01`–`q15`, `u01`–`u02`).

---

## 1. Chấm Context trên 5 câu Test có đáp án (0 / 1 / 2)

| Query ID | Gold chunk (trang) | Baseline (`test-tfidf`, budget=800) | Điểm Baseline | Cấu hình chọn (`test-hybrid-rerank-b3600`) | Điểm Chosen | Lý do đối chiếu |
|---|---|---|---:|---|---:|---|
| `q16` | `p33_para_0` (tr.33) | `['p33_para_0']` | **2** | `['p33_para_0', 'p20_para_3', 'p19_para_1']` | **2** | Cả hai đều giữ trọn `p33_para_0` có đủ câu của Buffett. |
| `q17` | `p29_para_0` (tr.29) | `['p29_para_0']` | **2** | `['p29_para_0', 'p29_para_2', 'p29_para_1']` | **2** | Cả hai đều giữ trọn `p29_para_0` có đủ chi tiết 250% và vụ lừa đảo tháng 10. |
| `q18` | `p1_para_6` (tr.1) | `['p1_para_6']` | **2** | `['p1_para_6', 'p1_para_2', 'p1_para_1']` | **2** | Cả hai đều giữ trọn `p1_para_6` có chi phí môi giới 8%. |
| `q19` | `p4_para_3` (tr.4) | `['p4_para_3']` | **2** | `['p4_para_3', 'p17_para_3', 'p21_para_3']` | **2** | Cả hai đều giữ trọn `p4_para_3` có số liệu 1973 (20%) và 2002 (105%). |
| `q20` | `p14_para_1` (tr.14) | `['p1_para_3']` | **0** | `['p14_para_1', 'p16_para_0', 'p23_para_2']` | **2** | Ở Baseline, `p14_para_1` đứng rank #1 nhưng cộng header vượt 800 ký tự nên bị loại khỏi context; Chosen giữ trọn `p14_para_1`. |
| **Tổng hợp** | | **Mức 0: 1, Mức 1: 0, Mức 2: 4** | **TB 1.60/2** | **Mức 0: 0, Mức 1: 0, Mức 2: 5** | **TB 2.00/2** | |

---

## 2. Chấm Agent trên 7 câu Test (Baseline vs Cấu hình đã chọn)

| Query ID | Loại | Baseline (`test-agent-baseline`) | Nhãn Baseline | Cấu hình chọn (`test-agent-chosen`) | Nhãn Chosen |
|---|---|---|---|---|---|
| `q16` | Có đáp án | Answered + `[p33_para_0]` | Grade: `2`, Cite: `[True]` | Answered + `[p33_para_0]` | Grade: `2`, Cite: `[True]` |
| `q17` | Có đáp án | Answered + `[p29_para_0]` | Grade: `2`, Cite: `[True]` | Answered + `[p29_para_0]` | Grade: `2`, Cite: `[True]` |
| `q18` | Có đáp án | Answered + `[p1_para_6]` | Grade: `2`, Cite: `[True]` | Answered + `[p1_para_6]` | Grade: `2`, Cite: `[True]` |
| `q19` | Có đáp án | Answered + `[p4_para_3]` | Grade: `2`, Cite: `[True]` | Answered + `[p4_para_3]` | Grade: `2`, Cite: `[True]` |
| `q20` | Có đáp án | **Abstain** (rớt `p14_para_1` do budget=800) | Grade: `0`, Cite: `[]` | Answered + `[p14_para_1]` | Grade: `2`, Cite: `[True]` |
| `u03` | Thiếu bằng chứng | Abstain (1 lượt search) | Abstain: `True` | Abstain (1 lượt search) | Abstain: `True` |
| `u04` | Thiếu bằng chứng | Abstain (2 lượt search) | Abstain: `True` | Abstain (3 lượt search) | Abstain: `True` |

