"""
backend/app.py  (Day 3-5 version)
/api/chat routes through the LangGraph registry Planner
(guardrail -> supervisor -> the chosen agent NODE).
"""
import os
import sys
from fastapi.responses import FileResponse

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
from agents.planner_agent.planner import Planner          # LangGraph registry planner

app = FastAPI(title="AI Legal Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

init_db()
EXTRACT_DIR = os.path.join("storage", "extracted_text")
os.makedirs(EXTRACT_DIR, exist_ok=True)

rag = RAGPipeline()
try:
    rag._get_knowledge_retriever()
except Exception as e:
    print("CUAD load will happen on first question:", e)

planner = Planner(rag)          # guardrail -> supervisor -> agent node
SESSION = {"has_document": False}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    result = validate_and_save(file.filename, file.content_type, contents)
    if result["status"] == "error":
        return result
    file_id = result["file_id"]
    saved_path = result["saved_path"]

    text = extract_text(saved_path)
    text_path = os.path.join(EXTRACT_DIR, f"{file_id}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)

    cls = classify(text)
    add_document(file_id, result["file_name"], cls["document_type"], cls["confidence"], text_path)
    n_chunks = rag.ingest_document_text(text, source_name=result["file_name"])
    SESSION["has_document"] = True

    return {
        "status": "success", "file_id": file_id, "file_name": result["file_name"],
        "document_type": cls["document_type"], "confidence": cls["confidence"],
        "method": cls["method"], "word_count": len(text.split()), "chunks": n_chunks,
    }


class ChatRequest(BaseModel):
    question: str
    general: bool = False

@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        result = planner.route(req.question,
                               has_document=SESSION["has_document"], general=req.general)
        return {"status": "success", "agent": result.get("agent", "Legal Q&A Agent"),
                "answer": result["answer"], "mode": result.get("mode", "knowledge")}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "answer": f"Error: {str(e)}", "mode": "error"}


@app.post("/api/reset")
def reset():
    SESSION["has_document"] = False
    rag.clear_memory()
    return {"status": "success", "message": "Cleared. Now answering from CUAD knowledge base."}


@app.get("/api/files")
def list_files():
    return {"status": "success", "files": get_all_documents()}

@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "document" if SESSION["has_document"] else "knowledge (CUAD)"}


FRONTEND = os.path.join(ROOT, "frontend")
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND, "js")), name="js")

@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND, "index.html"))

@app.get("/chat.html")
def chat_page():
    return FileResponse(os.path.join(FRONTEND, "chat.html"))

# ----------------------Serve/download generated report
@app.get("/api/report")
def get_report():
    import os
    path = os.path.join("storage", "generated_reports", "legal_report.pdf")
    if os.path.exists(path):
        return FileResponse(path, media_type="application/pdf", filename="legal_report.pdf")
    return {"status": "error", "message": "No report generated yet. Ask the chatbot to 'generate a report' first."}


if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
