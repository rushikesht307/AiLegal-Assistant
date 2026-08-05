import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from backend.services.file_handler import validate_and_save
from agents.ocr_agent.ocr_agent import extract_text
from agents.classification_agent.classifier import classify
from database.db import init_db
from database.crud import add_document, get_all_documents
from rag.pipeline import RAGPipeline

app = FastAPI(title="AI Legal Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

init_db()
EXTRACT_DIR = os.path.join("storage", "extracted_text")
os.makedirs(EXTRACT_DIR, exist_ok=True)

# ---- RAG pipeline (one instance for the whole app) ----
rag = RAGPipeline()
try:
    rag._get_knowledge_retriever()      # load CUAD knowledge base once on startup
except Exception as e:
    print("CUAD load will happen on first question:", e)

# session: is a document currently uploaded?
SESSION = {"has_document": False}


# ---------------- Upload: Day-1 pipeline + Day-2 RAG ingest ----------------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()

    # 1. validate & save (Adarsh)
    result = validate_and_save(file.filename, file.content_type, contents)
    if result["status"] == "error":
        return result
    file_id = result["file_id"]
    saved_path = result["saved_path"]

    # 2. extract text (Vasanth) - handles image/pdf/docx
    text = extract_text(saved_path)

    # save cleaned text
    text_path = os.path.join(EXTRACT_DIR, f"{file_id}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)

    # 3. hybrid classify (Alankrutha)
    cls = classify(text)

    # 4. store metadata (Meghana)
    add_document(file_id, result["file_name"], cls["document_type"], cls["confidence"], text_path)

    # 5. Day-2: ingest the OCR text into RAG (handles scanned docs too)
    n_chunks = rag.ingest_document_text(text, source_name=result["file_name"])
    SESSION["has_document"] = True

    # 6. return result
    return {
        "status": "success",
        "file_id": file_id,
        "file_name": result["file_name"],
        "document_type": cls["document_type"],
        "confidence": cls["confidence"],
        "method": cls["method"],
        "word_count": len(text.split()),
        "chunks": n_chunks,
    }


# ---------------- Chat: the 3 use cases via RAG ----------------
class ChatRequest(BaseModel):
    question: str
    general: bool = False     # set True for a general legal question

@app.post("/api/chat")
def chat(req: ChatRequest):
    result = rag.get_answer(
        req.question,
        has_document=SESSION["has_document"],
        general=req.general,
    )
    return {
        "status": "success",
        "answer": result["answer"],
        "mode": result["mode"],   # "document" or "knowledge"
    }


# ---------------- Reset: clear the document, back to CUAD mode ----------------
@app.post("/api/reset")
def reset():
    SESSION["has_document"] = False
    rag.clear_memory()
    return {"status": "success", "message": "Cleared. Now answering from CUAD knowledge base."}


# ---------------- List stored documents ----------------
@app.get("/api/files")
def list_files():
    return {"status": "success", "files": get_all_documents()}

@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "document" if SESSION["has_document"] else "knowledge (CUAD)"}


# ---------------- Serve the frontend ----------------
FRONTEND = os.path.join(ROOT, "frontend")
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND, "js")), name="js")

@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND, "index.html"))


# ---------------- Run ----------------
if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
 