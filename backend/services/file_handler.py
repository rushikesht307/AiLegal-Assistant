import os
import uuid

# Where valid files are stored
UPLOAD_DIR = os.path.join("storage", "uploaded_documents")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Only these types are allowed
ALLOWED_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tiff",
    "text/plain" : ".txt"
}

MAX_SIZE = 50 * 1024 * 1024  # 50 MB


def validate_and_save(filename: str, content_type: str, contents: bytes):
    """
    Returns a dict:
      success -> { "status": "success", "file_id", "saved_path", "file_name" }
      error   -> { "status": "error", "message" }
    """

    # 1. check the type
    if content_type not in ALLOWED_TYPES:
        return {"status": "error", "message": "Unsupported file type. Allowed: PDF, DOCX, PNG, JPG, TIFF."}

    # 2. check empty
    if len(contents) == 0:
        return {"status": "error", "message": "Uploaded file is empty."}

    # 3. check size
    if len(contents) > MAX_SIZE:
        return {"status": "error", "message": "File too large. Max 50MB allowed."}

    # 4. build a unique name so files don't overwrite each other
    file_id = str(uuid.uuid4())
    unique_name = f"{file_id}_{filename}"
    saved_path = os.path.join(UPLOAD_DIR, unique_name)

    # 5. save the file
    with open(saved_path, "wb") as f:
        f.write(contents)

    return {
        "status": "success",
        "file_id": file_id,
        "file_name": filename,
        "saved_path": saved_path,
    }
