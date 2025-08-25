from pathlib import Path
import subprocess
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
from pydantic import BaseModel
from text_extraction import read_upload_file
import shutil
import os
from dotenv import load_dotenv
from enum import Enum

load_dotenv()  # Load environment variables from .env file

# Import your classify_document and call_llm logic
from classifier import classify_document, DB_PATH, EMBED_MODEL

REFERENCE_PATH = os.getenv("REFERENCE_PATH", "data/reference")
class TextRequest(BaseModel):
    text: str

# -----------------------
# Initialize FastAPI
# -----------------------
app = FastAPI(title="ASA Document Classifier")


# -----------------------
# FastAPI Endpoint
# -----------------------
class TextRequest(BaseModel):
    text: str


class SourceFilter(str, Enum):
    asa_rrds = "asa_rrds"
    reference = "reference"

# Case 1: JSON text input
@app.post("/api/classify/text")
async def classify_text(req: TextRequest, source_filter: SourceFilter = None):
    result = classify_document(req.text, source_filter=source_filter)
    return result

# Case 2: File upload (TXT or PDF)
@app.post("/api/classify/file")
async def classify_file(file: UploadFile = File(...), source_filter: SourceFilter = None):
    if not file:
        return {"error": "No file provided."}
    
    try:
        text_to_classify = read_upload_file(file)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}

    result = classify_document(
        text_to_classify,
        source_filter=source_filter
    )
    return result

@app.post("/api/upload-sample", include_in_schema=False)
async def upload_sample(code: str = Form(...), file: UploadFile = None):
    code_dir = os.path.join(REFERENCE_PATH, code)
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)
    file_path = os.path.join(code_dir, file.filename)
    i = 1
    while True:
        if not os.path.exists(file_path):
            break
        file_path = os.path.join(code_dir, f"{os.path.splitext(file.filename)[0]}_{i}{os.path.splitext(file.filename)[1]}")
        i += 1
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": f"File saved to {file_path}"}

@app.post("/api/regenerate", include_in_schema=False)
async def regenerate_embeddings():
    try:
        subprocess.run(["python", "embedder.py"], check=True)
        return {"message": "Embeddings regenerated successfully."}
    except subprocess.CalledProcessError as e:
        return {"message": f"Error running embedder.py: {e}"}


# Serve the HTML file at root URL
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    with open("html/classify.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/trainer", response_class=HTMLResponse, include_in_schema=False)
async def trainer():
    with open("html/trainer.html", "r", encoding="utf-8") as f:
        return f.read()

# @app.get("/files/{asa_code}")
# async def list_files(asa_code: str):
#     files = get_files_for_code(asa_code)  # however you track them (disk, DB, S3, etc.)
#     return {"asa_code": asa_code, "files": files}