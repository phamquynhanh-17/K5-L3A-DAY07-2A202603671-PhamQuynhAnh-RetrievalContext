"""Save human-verified audit reviews (based on direct trace inspection) and run starter/agent_metrics.py."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REVIEWS = {
    "dev-agent-baseline": [
        {"query_id": "q01", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng năm 1950 và tuổi 19; p33_para_0 hỗ trợ trực tiếp."},
        {"query_id": "q02", "answer_grade": 0, "citation_support": [], "reason": "Từ chối nhầm câu có đáp án vì budget=800 bỏ p33_para_1."},
        {"query_id": "q03", "answer_grade": 1, "citation_support": [True], "reason": "Nêu đúng tuổi gần tám mươi từ p32_para_0 nhưng bỏ sót 3 điều mong muốn mỗi ngày."},
        {"query_id": "q04", "answer_grade": 0, "citation_support": [], "reason": "Từ chối nhầm câu có đáp án vì TF-IDF/budget=800 bỏ p30_para_4."},
        {"query_id": "q05", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng ít nhất 14,7%/năm; p29_para_2 hỗ trợ trực tiếp."},
        {"query_id": "q06", "answer_grade": 0, "citation_support": [], "reason": "Từ chối nhầm câu có đáp án vì budget=800 bỏ p29_para_4."},
        {"query_id": "q07", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng 80.000 đô la sau 20 năm; p27_para_3 hỗ trợ trực tiếp."},
        {"query_id": "q08", "answer_grade": 0, "citation_support": [], "reason": "Từ chối nhầm câu có đáp án vì TF-IDF không lấy được p20_para_1."},
        {"query_id": "q09", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng 20.000 bảng; p19_para_0 hỗ trợ trực tiếp."},
        {"query_id": "q10", "answer_grade": 0, "citation_support": [], "reason": "Từ chối nhầm câu có đáp án vì budget=800 bỏ p17_para_4 (rank 1)."},
        {"query_id": "q11", "answer_grade": 0, "citation_support": [], "reason": "Từ chối nhầm câu có đáp án vì budget=800 bỏ p16_para_1."},
        {"query_id": "q12", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng trên 90% và đủ hai lý do; p15_para_0 hỗ trợ trực tiếp."},
        {"query_id": "q13", "answer_grade": 1, "citation_support": [True], "reason": "Chỉ nêu tối thiểu 25% và 50-50 theo p25_para_5, thiếu giới hạn tối đa 75% ở p14_para_4."},
        {"query_id": "q14", "answer_grade": 1, "citation_support": [True], "reason": "Chỉ nêu mức 3,2% cuối năm 1964 từ p14_para_5, thiếu mức <3,5% lúc Dow=900 do budget=800 bỏ p13_para_2."},
        {"query_id": "q15", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng kỳ vọng tăng thêm ~5%; p6_para_2 hỗ trợ trực tiếp."},
        {"query_id": "u01", "abstention_correct": True, "citation_support": [], "reason": "Từ chối đúng câu thiếu nội dung Chương 8."},
        {"query_id": "u02", "abstention_correct": True, "citation_support": [], "reason": "Từ chối đúng câu thiếu nội dung Chương 20."},
    ],
    "dev-agent-hybrid-b2400": [
        # q01 had status=error (429 rate limit), excluded by agent_metrics.py
        {"query_id": "q02", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng Chương 8 và Chương 20; p33_para_1 hỗ trợ trực tiếp."},
        {"query_id": "q03", "answer_grade": 1, "citation_support": [True], "reason": "Giải thích đúng câu chữ về tuổi gần 80 từ p32_para_0 nhưng chưa nêu 3 điều mong muốn mỗi ngày."},
        {"query_id": "q04", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng vụ sụp đổ thị trường chứng khoán năm 1907; p30_para_4 hỗ trợ trực tiếp."},
        {"query_id": "q05", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng ít nhất 14,7%/năm; p29_para_2 hỗ trợ trực tiếp."},
        {"query_id": "q06", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đủ lý do tránh nguy cơ sai lầm và không mua quá đắt; p29_para_4 hỗ trợ trực tiếp."},
        {"query_id": "q07", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng 80.000 đô la sau 20 năm; p27_para_3 hỗ trợ trực tiếp."},
        {"query_id": "q08", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đủ 3 bài học mạnh mẽ; p20_para_1 hỗ trợ trực tiếp."},
        {"query_id": "q09", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng 20.000 bảng; p19_para_0 hỗ trợ trực tiếp."},
        {"query_id": "q10", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng điều kiện có đủ tiền chi tiêu trong tay; p17_para_4 hỗ trợ trực tiếp."},
        {"query_id": "q11", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đủ 3 tiêu chí năm 1934; p16_para_1 hỗ trợ trực tiếp."},
        {"query_id": "q12", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng trên 90% và đủ 2 lý do; p15_para_0 hỗ trợ trực tiếp."},
        {"query_id": "q13", "answer_grade": 2, "citation_support": [True, True], "reason": "Trả lời đúng khoảng 25%-75%; cả p14_para_4 và p11_para_3 đều hỗ trợ trực tiếp."},
        {"query_id": "q14", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đủ 3,2% (1964) và <3,5% (Dow 900); p13_para_2 hỗ trợ trực tiếp."},
        {"query_id": "q15", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng ~5%; p6_para_2 hỗ trợ trực tiếp."},
        {"query_id": "u01", "abstention_correct": False, "citation_support": [True, True], "reason": "Model từ chối về ngữ nghĩa nhưng vẫn gắn citation p33_para_1 và p17_para_4 nên status='answered'."},
        {"query_id": "u02", "abstention_correct": True, "citation_support": [], "reason": "Từ chối đúng câu thiếu nội dung Chương 20."},
    ],
    "dev-agent-hybrid-rerank-b3600": [
        {"query_id": "q01", "answer_grade": 2, "citation_support": [True], "reason": "Đúng năm 1950 và tuổi 19; p33_para_0 hỗ trợ trực tiếp."},
        {"query_id": "q02", "answer_grade": 2, "citation_support": [True], "reason": "Đúng Chương 8 và Chương 20; p33_para_1 hỗ trợ trực tiếp."},
        {"query_id": "q03", "answer_grade": 1, "citation_support": [True], "reason": "Đúng chi tiết gần 80 tuổi từ p32_para_0 nhưng thiếu 3 điều mong muốn mỗi ngày."},
        {"query_id": "q04", "answer_grade": 2, "citation_support": [True], "reason": "Đúng sự kiện sụp đổ thị trường chứng khoán năm 1907; p30_para_4 hỗ trợ trực tiếp."},
        {"query_id": "q05", "answer_grade": 2, "citation_support": [True], "reason": "Đúng ít nhất 14,7%/năm; p29_para_2 hỗ trợ trực tiếp."},
        {"query_id": "q06", "answer_grade": 2, "citation_support": [True], "reason": "Đủ ý nguy cơ sai lầm và không mua quá đắt; p29_para_4 hỗ trợ trực tiếp."},
        {"query_id": "q07", "answer_grade": 2, "citation_support": [True], "reason": "Đúng 80.000 đô la trong 20 năm; p27_para_3 hỗ trợ trực tiếp."},
        {"query_id": "q08", "answer_grade": 2, "citation_support": [True], "reason": "Đủ 3 bài học mạnh mẽ; p20_para_1 hỗ trợ trực tiếp."},
        {"query_id": "q09", "answer_grade": 2, "citation_support": [True], "reason": "Đúng 20.000 bảng (>3 triệu USD); p19_para_0 hỗ trợ trực tiếp."},
        {"query_id": "q10", "answer_grade": 2, "citation_support": [True], "reason": "Đúng điều kiện đủ tiền chi tiêu trong tay; p17_para_4 hỗ trợ trực tiếp."},
        {"query_id": "q11", "answer_grade": 2, "citation_support": [True, True], "reason": "Đủ 3 tiêu chí; cả p16_para_1 và p5_para_1 đều hỗ trợ trực tiếp."},
        {"query_id": "q12", "answer_grade": 2, "citation_support": [True], "reason": "Đúng >90% và đủ 2 lý do; p15_para_0 hỗ trợ trực tiếp."},
        {"query_id": "q13", "answer_grade": 2, "citation_support": [True, True], "reason": "Đúng 25%-75%; cả p14_para_4 và p11_para_3 đều nêu rõ 25% và 75%."},
        {"query_id": "q14", "answer_grade": 2, "citation_support": [True], "reason": "Đủ 3,2% cuối 1964 và <3,5% lúc Dow 900; p13_para_2 hỗ trợ trực tiếp."},
        {"query_id": "q15", "answer_grade": 2, "citation_support": [True], "reason": "Đúng kỳ vọng tăng thêm ~5%; p6_para_2 hỗ trợ trực tiếp."},
        {"query_id": "u01", "abstention_correct": True, "citation_support": [], "reason": "Từ chối đúng câu thiếu nội dung Chương 8 (không còn dính lỗi citation)."},
        {"query_id": "u02", "abstention_correct": True, "citation_support": [], "reason": "Từ chối đúng câu thiếu nội dung Chương 20."},
    ],
    "test-agent-baseline": [
        {"query_id": "q16", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng không đòi hỏi IQ cao hay thông tin nội bộ; p33_para_0 hỗ trợ trực tiếp."},
        {"query_id": "q17", "answer_grade": 2, "citation_support": [True], "reason": "Đúng 250% ngày đầu và tháng 10 bị phát hiện lừa đảo, cổ phiếu vô giá trị; p29_para_0 hỗ trợ."},
        {"query_id": "q18", "answer_grade": 2, "citation_support": [True], "reason": "Đúng chi phí môi giới chứng khoán nhỏ ăn tới 8% và nhiều người mua tháng 12; p1_para_6 hỗ trợ."},
        {"query_id": "q19", "answer_grade": 2, "citation_support": [True], "reason": "Đúng mức luân chuyển 20% (1973) và 105% (2002); p4_para_3 hỗ trợ trực tiếp."},
        {"query_id": "q20", "answer_grade": 0, "citation_support": [], "reason": "Từ chối nhầm câu có đáp án vì budget=800 bỏ mất p14_para_1 dù TF-IDF rank #1."},
        {"query_id": "u03", "abstention_correct": True, "citation_support": [], "reason": "Từ chối đúng câu Phụ lục / năm 2008 không có trong tài liệu."},
        {"query_id": "u04", "abstention_correct": True, "citation_support": [], "reason": "Từ chối đúng câu Chương 14 không có trong trích đoạn."},
    ],
    "test-agent-chosen": [
        {"query_id": "q16", "answer_grade": 2, "citation_support": [True], "reason": "Trả lời đúng không đòi hỏi IQ cao hay thông tin nội bộ; p33_para_0 hỗ trợ trực tiếp."},
        {"query_id": "q17", "answer_grade": 2, "citation_support": [True], "reason": "Đúng 250% ngày đầu và tháng 10 bị phát hiện lừa đảo, cổ phiếu vô giá trị; p29_para_0 hỗ trợ."},
        {"query_id": "q18", "answer_grade": 2, "citation_support": [True], "reason": "Đúng lý do mua tháng 12 và chi phí môi giới 8% làm tiêu biến lợi nhuận; p1_para_6 hỗ trợ."},
        {"query_id": "q19", "answer_grade": 2, "citation_support": [True], "reason": "Đúng mức luân chuyển 20% (1973) và 105% (2002); p4_para_3 hỗ trợ trực tiếp."},
        {"query_id": "q20", "answer_grade": 2, "citation_support": [True], "reason": "Đủ ý quỹ riêng càng nhỏ càng tốt và không đổ thêm tiền khi thị trường lên; p14_para_1 hỗ trợ."},
        {"query_id": "u03", "abstention_correct": True, "citation_support": [], "reason": "Từ chối đúng câu Phụ lục / năm 2008 không có trong tài liệu."},
        {"query_id": "u04", "abstention_correct": True, "citation_support": [], "reason": "Agent tự tìm 3 lượt với từ khóa khác nhau rồi từ chối đúng câu Chương 14."},
    ],
}


def main() -> None:
    cases_path = ROOT / "starter" / "data" / "day6_c" / "agent_cases.json"
    for run_name, review_list in REVIEWS.items():
        run_dir = ROOT / "starter" / "results" / run_name
        rev_path = run_dir / "reviews.json"
        rev_path.write_text(json.dumps(review_list, ensure_ascii=False, indent=2), encoding="utf-8")
        split = "dev" if run_name.startswith("dev-") else "test"
        out_path = run_dir / "summary.json"
        print(f"=== {run_name} ({split}) ===")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "starter" / "agent_metrics.py"),
                "--cases", str(cases_path),
                "--traces", str(run_dir / "traces.json"),
                "--reviews", str(rev_path),
                "--split", split,
                "--output", str(out_path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

