/* AI Legal Assistant - Single Chatbot logic (Owner: Rushikesh Tambe)
   Connected to the Day-1 backend:  POST /api/upload
   The chatbot upload -> real pipeline (validate, OCR/parse, classify, store).
   Question answers are demo (Day-1). Real agents connect on Day 2+. */

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

/* ---- Upload inside the chat -> real backend pipeline ---- */
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
      docStatus.innerHTML = "&#9989; " + data.file_name;
      docStatus.classList.add("active");
      addBot("Classification Agent",
        `I've processed <b>${data.file_name}</b> through OCR/parsing and classified it as ` +
        `<b>${data.document_type}</b> (${Math.round(data.confidence * 100)}% confidence), ` +
        `with about <b>${data.word_count}</b> words. You can now ask me about it or request a report.`);
    } else {
      addBot("Planner Agent", "Upload failed: " + data.message);
    }
  } catch (err) {
    typing.remove();
    addBot("Planner Agent", "Could not reach the backend. Make sure the server is running (python backend/app.py).");
  }
});

/* ---- Send a question -> Planner Agent (demo routing on Day-1) ---- */
function handleSend() {
  const text = chatInput.value.trim();
  if (!text) return;
  removeWelcome();
  addUser(text);
  chatInput.value = "";
  chatInput.style.height = "auto";

  if (!documentUploaded && !/^(hi|hello|hey)\b/i.test(text)) {
    const t = showTyping();
    setTimeout(() => { t.remove(); addBot("Planner Agent",
      "Please upload a document first using the &#128206; button so I can analyze it."); }, 700);
    return;
  }

  const route = planner(text);
  const t = showTyping();
  setTimeout(() => { t.remove(); addBot(route.agent, route.answer); }, 800);
}

/* The PLANNER AGENT (demo, keyword-based). Real LangGraph version comes Day 2+. */
function planner(text) {
  const q = text.toLowerCase();
  if (q.includes("report") || q.includes("summary"))
    return { agent: "Report Generator Agent", answer: "A full report will be generated here once the report agent is built (Day 5)." };
  if (q.includes("risk") || q.includes("risky"))
    return { agent: "Risk Analysis Agent", answer: "The Risk Analysis Agent will flag risky clauses here (Day 4)." };
  if (q.includes("clause") || q.includes("termination"))
    return { agent: "Clause Extraction Agent", answer: "The Clause Extraction Agent will list key clauses here (Day 3)." };
  if (q.includes("type") || q.includes("classify"))
    return { agent: "Classification Agent", answer: "The document type was detected at upload time (shown in the top-right status)." };
  return { agent: "Legal RAG Agent", answer: "The RAG Q&A agent will answer from your document with citations (Day 2)." };
}

/* ---- helpers ---- */
function addUser(t){const d=document.createElement("div");d.className="msg user";
  d.innerHTML=`<div class="avatar">&#128100;</div><div class="bubble">${esc(t)}</div>`;chatInner.appendChild(d);scroll();}
function addUserFile(n,s){const d=document.createElement("div");d.className="msg user";
  d.innerHTML=`<div class="avatar">&#128100;</div><div class="bubble"><div class="file-chip"><div class="fi">DOC</div>
  <div><b>${n}</b><br><span style="font-size:12px;opacity:.85;">${s}</span></div></div></div>`;chatInner.appendChild(d);scroll();}
function addBot(agent,html){const d=document.createElement("div");d.className="msg bot";
  d.innerHTML=`<div class="avatar">&#9878;</div><div class="bubble"><span class="tag">${agent}</span><br>${html}</div>`;chatInner.appendChild(d);scroll();}
function showTyping(){const d=document.createElement("div");d.className="msg bot";
  d.innerHTML=`<div class="avatar">&#9878;</div><div class="bubble">typing...</div>`;chatInner.appendChild(d);scroll();return d;}
function quickAsk(t){chatInput.value=t;handleSend();}
function removeWelcome(){if(welcome)welcome.style.display="none";}
function scroll(){const w=document.querySelector(".chat-wrap");w.scrollTop=w.scrollHeight;}
function esc(s){const d=document.createElement("div");d.textContent=s;return d.innerHTML;}
