"""Web Chatbot UI for Day 7 Agentic RAG (Python standard library HTTP server + modern HTML/JS)."""
import argparse
import json
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "src"))

from agent_runner import GeminiController  # noqa: E402
from agentic import run_agent  # noqa: E402
from retrievers import Day6Retriever  # noqa: E402

DATA_DIR = ROOT / "starter" / "data" / "day6_c"
CHUNKS = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))
VECTORS = json.loads((DATA_DIR / "vectors.json").read_text(encoding="utf-8"))["vectors"]
CHUNKS_BY_ID = {c["id"]: c for c in CHUNKS}

_RETRIEVERS: dict[str, Day6Retriever] = {}


def get_retriever(strategy: str) -> Day6Retriever:
    if strategy not in _RETRIEVERS:
        _RETRIEVERS[strategy] = Day6Retriever(
            chunks=CHUNKS,
            vectors=VECTORS,
            strategy=strategy,
            query_cache_path=DATA_DIR / "query_emb_cache.json",
        )
    return _RETRIEVERS[strategy]


HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Intelligent Investor · Agentic RAG</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #070b14;
      --bg-grad: radial-gradient(circle at 15% 10%, rgba(56, 189, 248, 0.08), transparent 35%),
                 radial-gradient(circle at 85% 90%, rgba(129, 140, 248, 0.08), transparent 40%);
      --sidebar: #0d1424;
      --panel: #131c31;
      --panel-hover: #1a2540;
      --border: rgba(148, 163, 184, 0.14);
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --indigo: #818cf8;
      --green: #34d399;
      --amber: #fbbf24;
      --red: #f87171;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      background-image: var(--bg-grad);
      color: var(--text);
      height: 100vh;
      display: flex;
      overflow: hidden;
    }
    /* Sidebar */
    aside {
      width: 300px;
      background: var(--sidebar);
      border-right: 1px solid var(--border);
      padding: 20px 18px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      overflow-y: auto;
      flex-shrink: 0;
    }
    .logo {
      display: flex;
      align-items: center;
      gap: 11px;
    }
    .logo-icon {
      width: 38px;
      height: 38px;
      border-radius: 11px;
      background: linear-gradient(135deg, var(--accent), var(--indigo));
      display: grid;
      place-items: center;
      font-weight: 800;
      color: #070b14;
      font-size: 16px;
      box-shadow: 0 6px 20px rgba(56, 189, 248, 0.25);
    }
    .logo h1 {
      margin: 0;
      font-size: 15px;
      font-weight: 700;
      letter-spacing: -0.01em;
    }
    .logo p {
      margin: 2px 0 0;
      font-size: 11.5px;
      color: var(--muted);
    }
    .section-title {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      font-weight: 600;
      margin-bottom: 10px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .field {
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }
    .field-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    select, input[type="number"] {
      width: 100%;
      background: #090e1a;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 13px;
      outline: none;
      transition: border-color 0.15s;
    }
    select:focus, input[type="number"]:focus { border-color: var(--accent); }
    .sample-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .sample-btn {
      text-align: left;
      background: var(--panel);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 12.5px;
      line-height: 1.4;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .sample-btn:hover {
      background: var(--panel-hover);
      border-color: rgba(56, 189, 248, 0.45);
      transform: translateY(-1px);
    }
    .sample-tag {
      display: inline-block;
      font-size: 10px;
      font-weight: 600;
      color: var(--accent);
      margin-bottom: 3px;
    }
    .sample-tag.warn { color: var(--amber); }
    .clear-btn {
      margin-top: auto;
      background: transparent;
      border: 1px solid var(--border);
      color: var(--muted);
      border-radius: 9px;
      padding: 9px 12px;
      font-size: 12.5px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .clear-btn:hover { color: var(--text); border-color: var(--muted); }

    /* Main Workspace */
    .workspace {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
    }
    .topbar {
      padding: 14px 26px;
      border-bottom: 1px solid var(--border);
      background: rgba(13, 20, 36, 0.7);
      backdrop-filter: blur(10px);
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 13px;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      background: rgba(52, 211, 153, 0.1);
      border: 1px solid rgba(52, 211, 153, 0.28);
      color: var(--green);
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 11.5px;
      font-weight: 600;
    }
    .dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 8px var(--green);
    }
    main {
      flex: 1;
      overflow-y: auto;
      padding: 24px 28px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .chat-container {
      max-width: 820px;
      width: 100%;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .msg {
      display: flex;
      gap: 12px;
      animation: fadeIn 0.2s ease;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
    .msg.user { flex-direction: row-reverse; }
    .avatar {
      width: 32px;
      height: 32px;
      border-radius: 10px;
      display: grid;
      place-items: center;
      font-size: 12px;
      font-weight: 700;
      flex-shrink: 0;
      margin-top: 2px;
    }
    .user .avatar { background: linear-gradient(135deg, #0284c7, #2563eb); color: #fff; }
    .bot .avatar  { background: linear-gradient(135deg, rgba(56,189,248,0.2), rgba(129,140,248,0.2)); color: var(--accent); border: 1px solid rgba(56,189,248,0.3); }
    .msg-body {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-width: 82%;
    }
    .user .msg-body { align-items: flex-end; }
    .bubble {
      padding: 13px 17px;
      border-radius: 16px;
      line-height: 1.6;
      font-size: 14px;
      white-space: pre-wrap;
    }
    .user .bubble {
      background: linear-gradient(135deg, #0284c7, #1d4ed8);
      color: #fff;
      border-top-right-radius: 4px;
      box-shadow: 0 6px 20px rgba(2, 132, 199, 0.25);
    }
    .bot .bubble {
      background: var(--panel);
      border: 1px solid var(--border);
      border-top-left-radius: 4px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .cite-pill {
      display: inline-block;
      background: rgba(56, 189, 248, 0.14);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.35);
      border-radius: 6px;
      padding: 0 6px;
      margin: 0 2px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11.5px;
      font-weight: 600;
    }
    .meta {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 11.5px;
      color: var(--muted);
    }
    .badge {
      padding: 2px 8px;
      border-radius: 999px;
      font-weight: 600;
      font-size: 10.5px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .badge.answered { background: rgba(52,211,153,0.14); color: var(--green); border: 1px solid rgba(52,211,153,0.3); }
    .badge.abstain  { background: rgba(251,191,36,0.14); color: var(--amber); border: 1px solid rgba(251,191,36,0.3); }
    .badge.error    { background: rgba(248,113,113,0.14); color: var(--red);   border: 1px solid rgba(248,113,113,0.3); }
    details {
      background: rgba(19, 28, 49, 0.75);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 9px 13px;
      font-size: 12px;
      width: 100%;
    }
    summary {
      cursor: pointer;
      color: var(--accent);
      font-weight: 600;
      user-select: none;
    }
    .trace-item {
      margin-top: 9px;
      padding-top: 9px;
      border-top: 1px dashed var(--border);
      color: #cbd5e1;
      line-height: 1.5;
    }
    .cite-box {
      margin-top: 8px;
      padding: 10px 12px;
      background: #090e1a;
      border-radius: 8px;
      border-left: 3px solid var(--accent);
      font-size: 12.5px;
      color: #cbd5e1;
      line-height: 1.55;
    }
    /* Typing dots */
    .typing { display: inline-flex; gap: 5px; align-items: center; padding: 4px 2px; }
    .typing i {
      width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
      animation: bounce 1.2s infinite ease-in-out;
    }
    .typing i:nth-child(2) { animation-delay: 0.15s; }
    .typing i:nth-child(3) { animation-delay: 0.3s; }
    @keyframes bounce { 0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1.1); } }

    /* Input Bar */
    footer {
      padding: 16px 28px 20px;
      background: linear-gradient(to top, rgba(7,11,20,0.95), rgba(7,11,20,0.6), transparent);
    }
    form {
      max-width: 820px;
      margin: 0 auto;
      display: flex;
      gap: 10px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 6px 6px 6px 14px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
    form:focus-within { border-color: rgba(56, 189, 248, 0.55); }
    input[type="text"] {
      flex: 1;
      background: transparent;
      color: var(--text);
      border: none;
      font-size: 14px;
      outline: none;
    }
    button[type="submit"] {
      background: linear-gradient(135deg, var(--accent), var(--indigo));
      color: #070b14;
      font-weight: 700;
      border: none;
      border-radius: 10px;
      padding: 10px 18px;
      font-size: 13.5px;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.1s;
    }
    button[type="submit"]:hover { opacity: 0.92; }
    button[type="submit"]:active { transform: scale(0.98); }
    button[type="submit"]:disabled { opacity: 0.45; cursor: not-allowed; }
    @media (max-width: 820px) {
      aside { display: none; }
      main { padding: 16px; }
      footer { padding: 12px 16px; }
    }
  </style>
</head>
<body>
  <aside>
    <div class="logo">
      <div class="logo-icon">AI</div>
      <div>
        <h1>Agentic RAG</h1>
        <p>Nhà đầu tư thông minh</p>
      </div>
    </div>

    <div>
      <div class="section-title">Cấu hình Agent</div>
      <div class="card">
        <div class="field">
          <span>Chiến lược Retrieval</span>
          <select id="strategy">
            <option value="hybrid_rerank" selected>hybrid_rerank (Khuyên dùng)</option>
            <option value="hybrid_rrf">hybrid_rrf</option>
            <option value="dense">dense (3072-dim)</option>
            <option value="tfidf">tfidf (Baseline)</option>
          </select>
        </div>
        <div class="field-row">
          <div class="field">
            <span>Budget (ký tự)</span>
            <input id="budget" type="number" value="3600" min="400" max="8000" step="200" />
          </div>
          <div class="field">
            <span>Max search</span>
            <input id="maxSearches" type="number" value="4" min="1" max="6" />
          </div>
        </div>
      </div>
    </div>

    <div>
      <div class="section-title">Gợi ý câu hỏi</div>
      <div class="sample-list">
        <button class="sample-btn" onclick="askSample(this.dataset.q)" data-q="Theo lời nói đầu của Warren E. Buffett, hai chương nào trong sách cần được đặc biệt chú ý tới?">
          <span class="sample-tag">CÓ BẰNG CHỨNG</span><br/>Hai chương nào được Buffett khuyên đặc biệt chú ý?
        </button>
        <button class="sample-btn" onclick="askSample(this.dataset.q)" data-q="Isaac Newton đã mất bao nhiêu tiền khi quay lại mua cổ phiếu Công ty South Sea?">
          <span class="sample-tag">CÓ BẰNG CHỨNG</span><br/>Isaac Newton mất bao nhiêu tiền ở vụ South Sea?
        </button>
        <button class="sample-btn" onclick="askSample(this.dataset.q)" data-q="Theo Graham trong Chương 1, nếu người không chuyên muốn thử vận may đầu cơ thì cần tuân thủ nguyên tắc quản lý tiền vốn nào?">
          <span class="sample-tag">TEST q20</span><br/>Nguyên tắc quản lý vốn khi đầu cơ?
        </button>
        <button class="sample-btn" onclick="askSample(this.dataset.q)" data-q="Trong Chương 8 của cuốn sách, Benjamin Graham giải thích khái niệm 'Ngài Thị trường' (Mr. Market) như thế nào?">
          <span class="sample-tag warn">THIẾU BẰNG CHỨNG</span><br/>Giải thích khái niệm Ngài Thị trường (Chương 8)
        </button>
      </div>
    </div>

    <button class="clear-btn" onclick="clearChat()">Xóa lịch sử trò chuyện</button>
  </aside>

  <div class="workspace">
    <div class="topbar">
      <div style="color: var(--muted)">Corpus: <b style="color: var(--text)">171 chunks (Strategy C)</b> · Model: <b style="color: var(--text)">gemini-3.1-flash-lite</b></div>
      <div class="status-pill"><span class="dot"></span> Sẵn sàng</div>
    </div>

    <main id="chat">
      <div class="chat-container" id="chatList">
        <div class="msg bot">
          <div class="avatar">AI</div>
          <div class="msg-body">
            <div class="bubble">Xin chào! Mình là trợ lý <b>Agentic RAG</b>. Hãy đặt câu hỏi về trích đoạn sách <b>Nhà đầu tư thông minh</b> — mình sẽ tự tìm bằng chứng trong tài liệu, trả lời kèm trích dẫn <span class="cite-pill">[chunk_id]</span> hoặc chủ động từ chối nếu không đủ cơ sở.</div>
          </div>
        </div>
      </div>
    </main>

    <footer>
      <form id="chatForm">
        <input id="questionInput" type="text" placeholder="Đặt câu hỏi về cuốn Nhà đầu tư thông minh..." autocomplete="off" required />
        <button id="sendBtn" type="submit">Gửi</button>
      </form>
    </footer>
  </div>

  <script>
    const chat = document.getElementById("chat");
    const chatList = document.getElementById("chatList");
    const form = document.getElementById("chatForm");
    const input = document.getElementById("questionInput");
    const sendBtn = document.getElementById("sendBtn");

    function escapeHtml(str) {
      return String(str).replace(/[&<>"']/g, s => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[s]));
    }

    function formatAnswer(text) {
      return escapeHtml(text).replace(/\\[([a-zA-Z0-9_\\-]+)\\]/g, '<span class="cite-pill">[$1]</span>');
    }

    function scrollBottom() {
      chat.scrollTop = chat.scrollHeight;
    }

    function appendUser(text) {
      const div = document.createElement("div");
      div.className = "msg user";
      div.innerHTML = `
        <div class="avatar">Bạn</div>
        <div class="msg-body"><div class="bubble">${escapeHtml(text)}</div></div>
      `;
      chatList.appendChild(div);
      scrollBottom();
    }

    function appendLoading() {
      const div = document.createElement("div");
      div.className = "msg bot";
      div.innerHTML = `
        <div class="avatar">AI</div>
        <div class="msg-body">
          <div class="bubble"><div class="typing"><i></i><i></i><i></i><span style="margin-left:8px;color:var(--muted);font-size:13px">Đang truy xuất & kiểm chứng...</span></div></div>
        </div>
      `;
      chatList.appendChild(div);
      scrollBottom();
      return div;
    }

    function renderBot(data) {
      const div = document.createElement("div");
      div.className = "msg bot";
      const status = data.status || "error";
      const badgeCls = status === "answered" ? "answered" : (status === "abstain" ? "abstain" : "error");

      let traceHtml = "";
      if (data.searches && data.searches.length) {
        traceHtml += data.searches.map((s, idx) => `
          <div class="trace-item">
            <b>Lượt ${idx + 1}:</b> <code>${escapeHtml(s.query)}</code><br/>
            <span>Top-3: ${s.top_ids.map(x => `<code>${escapeHtml(x)}</code>`).join(" · ")}</span>
          </div>
        `).join("");
      }
      if (data.cited_chunks && data.cited_chunks.length) {
        traceHtml += data.cited_chunks.map(c => `
          <div class="cite-box">
            <b>[${escapeHtml(c.id)}]</b> · Trang ${c.page} (${escapeHtml(c.source)})<br/>
            ${escapeHtml(c.text)}
          </div>
        `).join("");
      }

      const detailsBlock = traceHtml
        ? `<details><summary>Xem Trace (${(data.searches||[]).length} lượt tìm) & Nguồn trích dẫn (${(data.citations||[]).length})</summary>${traceHtml}</details>`
        : "";

      div.innerHTML = `
        <div class="avatar">AI</div>
        <div class="msg-body">
          <div class="bubble">${formatAnswer(data.answer || data.error || "Có lỗi xảy ra.")}</div>
          <div class="meta">
            <span class="badge ${badgeCls}">${escapeHtml(status)}</span>
            <span>${escapeHtml(data.strategy)}</span>
            <span>· ${data.elapsed_ms} ms</span>
            <span>· ${data.tokens} tokens</span>
          </div>
          ${detailsBlock}
        </div>
      `;
      chatList.appendChild(div);
      scrollBottom();
    }

    function clearChat() {
      chatList.innerHTML = `
        <div class="msg bot">
          <div class="avatar">AI</div>
          <div class="msg-body">
            <div class="bubble">Đã làm mới cuộc trò chuyện. Hãy nhập câu hỏi mới của bạn!</div>
          </div>
        </div>
      `;
    }

    async function sendQuestion(q) {
      if (!q.trim()) return;
      appendUser(q);
      input.value = "";
      sendBtn.disabled = true;
      const loadingEl = appendLoading();
      try {
        const resp = await fetch("/api/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            question: q,
            strategy: document.getElementById("strategy").value,
            budget: Number(document.getElementById("budget").value),
            max_searches: Number(document.getElementById("maxSearches").value),
          }),
        });
        const data = await resp.json();
        loadingEl.remove();
        renderBot(data);
      } catch (err) {
        loadingEl.remove();
        renderBot({status: "error", answer: "Lỗi kết nối server: " + err.message, strategy: "-", elapsed_ms: 0, tokens: 0});
      } finally {
        sendBtn.disabled = false;
        input.focus();
      }
    }

    function askSample(q) {
      if (sendBtn.disabled) return;
      sendQuestion(q);
    }

    form.addEventListener("submit", (e) => {
      e.preventDefault();
      sendQuestion(input.value);
    });
  </script>
</body>
</html>
"""


class ChatbotHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path == "/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "chunks": len(CHUNKS)})
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            question = str(body.get("question", "")).strip()
            strategy = str(body.get("strategy", "hybrid_rerank"))
            budget = int(body.get("budget", 3600))
            max_searches = int(body.get("max_searches", 4))
            if not question:
                self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "error": "Câu hỏi không được rỗng"})
                return

            retriever = get_retriever(strategy)
            controller = GeminiController()
            t0 = time.perf_counter()
            tok0 = retriever.rerank_tokens
            res = run_agent(
                question=question,
                controller=controller,
                retriever=retriever,
                max_searches=max_searches,
                top_k=3,
                budget=budget,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            tokens = controller.total_tokens + (retriever.rerank_tokens - tok0)

            searches = [
                {
                    "query": e["query"],
                    "top_ids": [f"{r['id']} (p.{r['page']})" for r in e["results"]],
                    "context_ids": e["context_ids"],
                }
                for e in res["events"]
                if e.get("event") == "search"
            ]
            cited_chunks = [
                {
                    "id": cid,
                    "page": CHUNKS_BY_ID[cid]["page"],
                    "source": CHUNKS_BY_ID[cid]["source"],
                    "text": CHUNKS_BY_ID[cid]["text"],
                }
                for cid in res.get("citations", [])
                if cid in CHUNKS_BY_ID
            ]
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": res["status"],
                    "answer": res["answer"],
                    "citations": res["citations"],
                    "searches": searches,
                    "cited_chunks": cited_chunks,
                    "strategy": strategy,
                    "elapsed_ms": elapsed_ms,
                    "tokens": tokens,
                },
            )
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"status": "error", "error": str(exc), "answer": f"Lỗi: {exc}", "citations": [], "searches": [], "cited_chunks": []},
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Day 7 Agentic RAG Web Chatbot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ChatbotHandler)
    print(f"Chatbot server running at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

