const API = "http://127.0.0.1:8000/api";

// go to the chatbot page
function goChat() {
  window.location.href = "chat.html";
}

// tile clicked -> open chat and auto-ask the question
function askOnChat(question) {
  localStorage.setItem("presetQuestion", question);   // chat.js picks this up
  window.location.href = "chat.html";
}

// load uploaded documents into the home dashboard (if the panel exists)
async function loadDocs() {
  const docList = document.getElementById("docList");
  if (!docList) return;                                // no dashboard on this page
  try {
    const res = await fetch(`${API}/files`);
    const data = await res.json();
    if (data.files && data.files.length) {
      docList.innerHTML = "";
      data.files.forEach((f) => {
        docList.innerHTML += `
          <div class="doc-item">
            <div class="di">DOC</div>
            <div class="meta">
              <div class="nm">${f.file_name}</div>
              <div class="sb">${f.upload_time || ""}</div>
            </div>
            <span class="tag-pill">${f.document_type || "Document"}</span>
          </div>`;
      });
    }
  } catch (err) {
    // keep the empty state if backend not reachable
  }
}

// upload a document from the home page
const homeFile = document.getElementById("homeFile");
if (homeFile) {
  homeFile.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch(`${API}/upload`, { method: "POST", body: fd });
      const d = await res.json();
      if (d.status === "success") {
        await loadDocs();                 // reload the doc list
        window.location.href = "chat.html";
      } else {
        alert("Upload failed: " + d.message);
      }
    } catch (err) {
      alert("Could not reach the backend. Make sure the server is running.");
    }
  });
}

loadDocs();