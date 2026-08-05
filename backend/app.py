import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from backend.services.file_handler import validate_and_save    
from agents.ocr_agent.ocr_agent import extract_text            
from agents.classification_agent.classifier import classify      
from database.db import init_db                                  
from database.crud import add_document, get_all_documents        

app = FastAPI(title="AI Legal Assistant - Day 1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

init_db()
EXTRACT_DIR = os.path.join("storage", "extracted_text")
os.makedirs(EXTRACT_DIR, exist_ok=True)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()

    # 1. validate & save (Adarsh)
    result = validate_and_save(file.filename, file.content_type, contents)
    if result["status"] == "error":
        return result
    file_id = result["file_id"]
    saved_path = result["saved_path"]

    # 2. extract text (Vasanth) - handles image/pdf; docx returns "" for now
    text = extract_text(saved_path)

    # save cleaned text
    text_path = os.path.join(EXTRACT_DIR, f"{file_id}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)

    # 3. hybrid classify (Alankrutha)
    cls = classify(text)

    # 4. store metadata (Meghana)
    add_document(file_id, result["file_name"], cls["document_type"], cls["confidence"], text_path)

    # 5. return result
    return {
        "status": "success",
        "file_id": file_id,
        "file_name": result["file_name"],
        "document_type": cls["document_type"],
        "confidence": cls["confidence"],
        "method": cls["method"],
        "word_count": len(text.split()),
    }


@app.get("/api/files")
def list_files():
    return {"status": "success", "files": get_all_documents()}

@app.get("/api/health")
def health():
    return {"status": "ok", "message": "AI Legal Assistant Day-1 backend running"}


FRONTEND = os.path.join(ROOT, "frontend")
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND, "js")), name="js")

@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND, "index.html"))


if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
 