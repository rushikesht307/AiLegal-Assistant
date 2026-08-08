/* AI Legal Assistant - ChatGPT-style chat logic (Owner: Rushikesh Tambe)
   Connected to backend:
     POST /api/upload -> validate, OCR/parse, classify, store, ingest into RAG
     POST /api/chat   -> Planner routes to the right agent (document / CUAD)
     POST /api/reset  -> clear the document, back to CUAD mode
     GET  /api/files  -> list uploaded documents (sidebar)
     GET  /api/report -> download the generated PDF report
*/

const API = "http://127.0.0.1:8000/api";

const chatInner = document.getElementById("chatInner");
const chatScroll = document.getElementById("chatScroll");
const chatInput  = document.getElementById("chatInput");
const fileInput  = document.getElementById("fileInput");
const chatEmpty  = document.getElementById("chatEmpty");
const modePill   = document.getElementById("modePill");
const sideDocs   = document.getElementById("sideDocs");
const sendBtn    = document.getElementById("sendBtn");

let hasDocument = false;

/* ---- markdown -> HTML (renders **bold**, * bullets, etc.) ---- */
function renderMarkdown(text) {
  if (window.marked) {
    return marked.parse(text);          // full markdown if marked.js is loaded
  }
  // simple fallback if marked.js isn't available
  return text
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/^\s*\*\s+/gm, "&bull; ")
    .replace(/\n/g, "br");
}

/* auto-grow textarea */
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + "px";
});
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
});

/* load the sidebar with the current document */
async function loadSidebar() {
  try {
    const res = await fetch(`${API}/files`);
    const data = await res.json();
    if (data.files && data.files.length) {
      const latest = data.files[0];
      hasDocument = true;
      sideDocs.innerHTML =
        `<div class="side-item"><span class="si">&#128196;</span><span class="st">${latest.file_name}</span></div>`;
      modePill.innerHTML = "&#128196; Document Mode";
      modePill.classList.add("doc");
    }
  } catch (err) { /* backend not reachable - keep default */ }
}
loadSidebar();

/* auto-ask a preset question that came from a home-page tile */
window.addEventListener("DOMContentLoaded", () => {
  const preset = localStorage.getItem("presetQuestion");
  if (preset) {
    localStorage.removeItem("presetQuestion");
    chatInput.value = preset;
    handleSend();
  }
});

/* new chat -> reset */
async function newChat() {
  try { await fetch(`${API}/reset`, { method: "POST" }); } catch (e) {}
  hasDocument = false;
  modePill.innerHTML = "&#128218; Knowledge Mode";
  modePill.classList.remove("doc");
  sideDocs.innerHTML =
    `<div class="side-item" style="color:var(--muted);"><span class="si">&#128196;</span><span class="st">No document uploaded</span></div>`;
  chatInner.innerHTML = "";
  location.reload();
}

/* upload a document inside the chat */
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  removeEmpty();
  addUserFile(file.name, (file.size / (1024 * 1024)).toFixed(2) + " MB");
  const typing = showTyping();

  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch(`${API}/upload`, { method: "POST", body: fd });
    const d = await res.json();
    typing.remove();
    if (d.status === "success") {
      hasDocument = true;
      modePill.innerHTML = "&#128196; Document Mode";
      modePill.classList.add("doc");
      sideDocs.innerHTML =
        `<div class="side-item"><span class="si">&#128196;</span><span class="st">${d.file_name}</span></div>`;
      addBot(
        `I've analyzed <b>${d.file_name}</b> and classified it as <b>${d.document_type}</b> ` +
        `(${Math.round(d.confidence * 100)}% confidence), ~${d.word_count} words` +
        `${d.chunks ? ", indexed " + d.chunks + " chunks" : ""}. Ask me anything about it!`,
        null);
    } else {
      addBot("Upload failed: " + d.message, null);
    }
  } catch (err) {
    typing.remove();
    addBot("Could not reach the backend. Make sure the server is running.", null);
  }
});

/* send a question */
async function handleSend() {
  const text = chatInput.value.trim();
  if (!text) return;
  removeEmpty();
  addUser(text);
  chatInput.value = "";
  chatInput.style.height = "auto";
  const typing = showTyping();

  try {
    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, general: false }),
    });
    const d = await res.json();
    typing.remove();
    if (d.status === "success") {
      const src = d.mode === "document"
        ? "Source: your uploaded document"
        : "Source: CUAD legal knowledge base";
      // show which agent answered (if provided) + render markdown
      const agentTag = d.agent ? `<div class="agent-tag">${d.agent}</div>` : "";
      const botRow = addBot(agentTag + renderMarkdown(d.answer), src);

      // if the Report agent answered, add a View / Download button
      if (d.agent && d.agent.toLowerCase().includes("report")) {
        const link = document.createElement("a");
        link.href = `${API}/report`;
        link.target = "_blank";
        link.className = "report-btn";
        link.innerHTML = "&#8681; View / Download Report";
        botRow.querySelector(".content").appendChild(link);
      }
    } else {
      addBot(d.answer || "Sorry, I couldn't get an answer.", null);
    }
  } catch (err) {
    typing.remove();
    addBot("Could not reach the backend. Make sure the server is running.", null);
  }
}

/* ---- message builders (ChatGPT style rows) ---- */
function addUser(text) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `<div class="avatar">&#128100;</div>
    <div class="content"><div class="role">You</div><div class="text">${esc(text)}</div></div>`;
  chatInner.appendChild(row); scrollDown();
}
function addUserFile(name, size) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `<div class="avatar">&#128100;</div>
    <div class="content"><div class="role">You</div>
    <div class="file-chip"><div class="fi">DOC</div><div><b>${name}</b>

    <span style="font-size:12px;color:var(--muted);">${size}</span></div></div></div>`;
  chatInner.appendChild(row); scrollDown();
}
function addBot(html, src) {
  const row = document.createElement("div");
  row.className = "msg-row bot";
  let inner = `<div class="role">AI Legal Assistant</div><div class="text">${html}</div>`;
  if (src) inner += `<div class="src">${src}</div>`;
  row.innerHTML = `<div class="avatar">&#9878;</div><div class="content">${inner}</div>`;
  chatInner.appendChild(row); scrollDown();
  return row;                 // returns the row (needed for the report button)
}
function showTyping() {
  const row = document.createElement("div");
  row.className = "msg-row bot";
  row.innerHTML = `<div class="avatar">&#9878;</div>
    <div class="content"><div class="role">AI Legal Assistant</div>
    <div class="typing-dots"><span><span></span><span></span></div></div>`;
  chatInner.appendChild(row); scrollDown(); return row;
}

function quickAsk(t){ chatInput.value = t; handleSend(); }
function removeEmpty(){ if (chatEmpty) chatEmpty.style.display = "none"; }
function scrollDown(){ chatScroll.scrollTop = chatScroll.scrollHeight; }
function esc(s){ const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
 