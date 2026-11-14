"""Print agent traces alongside expected answers and cited chunk texts for human audit."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Result folder under starter/results/")
    args = parser.parse_args()

    cases = {
        c["id"]: c
        for c in json.loads((ROOT / "starter" / "data" / "day6_c" / "agent_cases.json").read_text(encoding="utf-8"))
    }
    traces = json.loads((ROOT / "starter" / "results" / args.run / "traces.json").read_text(encoding="utf-8"))
    for t in traces:
        qid = t["query_id"]
        c = cases[qid]
        searches = [e for e in t["events"] if e.get("event") == "search"]
        print(f"=== {qid} | status={t['status']} | searches={len(searches)} | ms={t['elapsed_ms']} | tok={t['tokens']} ===")
        print("Q:", c["question"])
        if c["answerable"]:
            print("EXPECTED:", c["expected_answer"])
            print("GOLD:", c["gold"])
        else:
            print("ABSENCE:", c["absence_reason"])
        print("ANSWER:", t["answer"])
        print("CITES:", t["citations"])
        for cid in t["citations"]:
            found = [r["text"] for s in searches for r in s["results"] if r["id"] == cid]
            snippet = found[0].replace("\n", " ")[:400] if found else "NOT_FOUND"
            print(f"  CITED[{cid}]: {snippet}")
        if searches:
            print("  LAST_CTX_IDS:", searches[-1].get("context_ids", []))


if __name__ == "__main__":
    main()

