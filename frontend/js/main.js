/* AI Legal Assistant - Single Chatbot logic (Owner: Rushikesh Tambe)
   Connected to the Day-2 backend:
     POST /api/upload -> validate, OCR/parse, classify, store, ingest into RAG
     POST /api/chat   -> RAG answers (3 use cases: document / CUAD)
     POST /api/reset  -> clear the document, back to CUAD mode
   Two-mode:
     - no document uploaded  -> answers from CUAD
     - document uploaded     -> answers from the document
*/

const API = "http://127.0.0.1:8000/api";

const chatInner = document.getElementById("chatInner");
const chatInput = document.getElementById("chatInput");
const fileInput = document.getElementById("fileInput");
const welcome   = document.getElementById("welcome");
const docStatus = document.getElementById("docStatus");

let documentUploaded = false;

chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
});
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
});

/* ---- Upload inside the chat -> real backend pipeline + RAG ingest ---- */
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  removeWelcome();
  addUserFile(file.name, (file.size / (1024 * 1024)).toFixed(2) + " MB");

  const typing = showTyping();

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API}/upload`, { method: "POST", body: formData });
    const data = await res.json();
    typing.remove();

    if (data.status === "success") {
      documentUploaded = true;
      docStatus.innerHTML = "&#128196; " + data.file_name;
      docStatus.classList.add("active");
      addBot("Classification Agent",
        `I've processed <b>${data.file_name}</b> through OCR/parsing and classified it as ` +
        `<b>${data.document_type}</b> (${Math.round(data.confidence * 100)}% confidence), ` +
        `with about <b>${data.word_count}</b> words${data.chunks ? " and indexed " + data.chunks + " chunks" : ""}. ` +
        `You can now ask me anything about it.`);
    } else {
      addBot("Planner Agent", "Upload failed: " + data.message);
    }
  } catch (err) {
    typing.remove();
    addBot("Planner Agent", "Could not reach the backend. Make sure the server is running (python backend/app.py).");
  }
});

/* ---- Send a question -> real RAG backend (/api/chat) ---- */
async function handleSend() {
  const text = chatInput.value.trim();
  if (!text) return;
  removeWelcome();
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
    const data = await res.json();
    typing.remove();

    if (data.status === "success") {
      // show which source answered: the document or the CUAD knowledge base
      const agent = data.mode === "document" ? "Document (RAG)" : "CUAD Knowledge (RAG)";
      addBot(agent, esc(data.answer).replace(/\n/g, "<br>"),
             data.mode === "document" ? "Source: your uploaded document"
                                      : "Source: CUAD legal knowledge base");
    } else {
      addBot("Planner Agent", "Sorry, I couldn't get an answer.");
    }
  } catch (err) {
    typing.remove();
    addBot("Planner Agent", "Could not reach the backend. Make sure the server is running (python backend/app.py).");
  }
}

/* ---- helpers ---- */
function addUser(t){const d=document.createElement("div");d.className="msg user";
  d.innerHTML=`<div class="avatar">&#128100;</div><div class="bubble">${esc(t)}</div>`;chatInner.appendChild(d);scroll();}

function addUserFile(n,s){const d=document.createElement("div");d.className="msg user";
  d.innerHTML=`<div class="avatar">&#128100;</div><div class="bubble"><div class="file-chip"><div class="fi">DOC</div>
  <div><b>${n}</b><br><span style="font-size:12px;opacity:.85;">${s}</span></div></div></div>`;chatInner.appendChild(d);scroll();}

function addBot(agent,html,ref){const d=document.createElement("div");d.className="msg bot";
  let inner=`<span class="tag">${agent}</span><br>${html}`;
  if(ref)inner+=`<span class="ref">${ref}</span>`;
  d.innerHTML=`<div class="avatar">&#9878;</div><div class="bubble">${inner}</div>`;chatInner.appendChild(d);scroll();}

function showTyping(){const d=document.createElement("div");d.className="msg bot";
  d.innerHTML=`<div class="avatar">&#9878;</div><div class="bubble">typing...</div>`;chatInner.appendChild(d);scroll();return d;}

function quickAsk(t){chatInput.value=t;handleSend();}
function removeWelcome(){if(welcome)welcome.style.display="none";}
function scroll(){const w=document.querySelector(".chat-wrap");w.scrollTop=w.scrollHeight;}
function esc(s){const d=document.createElement("div");d.textContent=s;return d.innerHTML;}
