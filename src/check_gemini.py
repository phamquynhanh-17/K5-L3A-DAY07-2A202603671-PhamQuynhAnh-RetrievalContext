"""Verify Gemini API key, chat model (gemini-3.1-flash-lite), and embedding model (3072-dim) connectivity."""
import os
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent


def load_gemini_key() -> str:
    candidates = [
        ROOT / ".env",
        ROOT / "data" / "Lab_06" / "K5-L3-Day06-DataIngestionEngineering" / "starter" / ".env",
    ]
    for path in candidates:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip("\"'")
                    if val:
                        return val
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    return key


def gemini_generate(
    prompt: str,
    *,
    model: str = "gemini-3.1-flash-lite",
    system_instruction: str | None = None,
    json_mode: bool = False,
    timeout: int = 25,
) -> tuple[str, int]:
    """Call Gemini generateContent REST API with explicit timeout and return (text, total_tokens)."""
    api_key = load_gemini_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0},
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"

    for attempt in range(4):
        resp = requests.post(url, params={"key": api_key}, json=body, timeout=timeout)
        if resp.status_code == 429 and attempt < 3:
            time.sleep(25)
            continue
        if resp.status_code >= 500 and attempt < 3:
            time.sleep(3 * (attempt + 1))
            continue
        if not resp.ok:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if "text" in p)
        tokens = int(data.get("usageMetadata", {}).get("totalTokenCount", 0))
        return text, tokens
    raise RuntimeError("Gemini API failed after retries")


def main() -> None:
    text, tokens = gemini_generate("Reply OK", model="gemini-3.1-flash-lite")
    print("Chat OK:", text.strip(), "tokens:", tokens)

    api_key = load_gemini_key()
    emb_resp = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents",
        params={"key": api_key},
        json={"requests": [{"model": "models/gemini-embedding-001", "content": {"parts": [{"text": "test"}]}, "taskType": "RETRIEVAL_QUERY"}]},
        timeout=25,
    )
    emb_resp.raise_for_status()
    dim = len(emb_resp.json()["embeddings"][0]["values"])
    print("Embed dim:", dim)


if __name__ == "__main__":
    main()
