import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
from embedder import embed

load_dotenv()  # Load environment variables from .env file

# Startup checks
REFERENCE_PATH = os.getenv("REFERENCE_PATH", "data/reference")
ASA_DOC_PATH= os.getenv("ASA_DOC_PATH", "data/asa_rrds")

logging.info(f"Checking reference directory and asa_rrds.csv existence...")
if not os.path.exists(REFERENCE_PATH):
    logging.info(f"Creating reference directory at {REFERENCE_PATH}")
    os.makedirs(REFERENCE_PATH)
if not os.path.exists(ASA_DOC_PATH):
    logging.info(f"Copying asa_rrds.csv to {ASA_DOC_PATH}")
    shutil.copy("bak/asa_rrds.csv", ASA_DOC_PATH)

logging.info("Checking database directory and embeddings...")
db_path = os.getenv("DB_PATH", "db/chroma_asa_rrds")
if not os.path.exists(db_path):
    logging.info(f"Creating database directory at {db_path}")
    os.makedirs(db_path)
    logging.info(f"Embedding documents into database at {db_path}")
    embed(db_path=db_path)

# Import your classify_document and call_llm logic
from classifier import classify_document, DB_PATH, EMBED_MODEL


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
    logging.info(f"Classifying text with source filter: {source_filter}")
    result = classify_document(req.text, source_filter=source_filter)
    return result

# Case 2: File upload (TXT or PDF)
@app.post("/api/classify/file")
async def classify_file(file: UploadFile = File(...), source_filter: SourceFilter = None):
    logging.info(f"Classifying file: {file.filename} with source filter: {source_filter}")
    if not file:
        logging.warning("No file provided.")
        return {"error": "No file provided."}
    
    try:
        text_to_classify = read_upload_file(file)
        logging.debug(f"Extracted text: {text_to_classify[:200]}...")  # Log first 200 characters
    except ValueError as e:
        logging.error(f"Error reading file {file.filename}: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logging.error(f"Failed to read file {file.filename}: {str(e)}")
        return {"error": f"Failed to read file: {str(e)}"}

    logging.debug(f"Calling classify_document")
    result = classify_document(
        text_to_classify,
        source_filter=source_filter
    )
    logging.debug(f"Classification result: {result}")  # Log first 200 characters
    return result

@app.post("/api/upload-sample", include_in_schema=False)
async def upload_sample(code: str = Form(...), file: UploadFile = None):
    logging.info(f"Uploading sample for code: {code}")
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
        logging.info(f"Saving uploaded file to {file_path}")
        shutil.copyfileobj(file.file, f)

    return {"message": f"File saved to {file_path}"}

@app.post("/api/regenerate", include_in_schema=False)
async def regenerate_embeddings():
    logging.info("Regenerating embeddings by calling embedder.py")
    try:
        subprocess.run(["python", "embedder.py"], check=True)
        logging.info("Embeddings regenerated successfully.")
        return {"message": "Embeddings regenerated successfully."}
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running embedder.py: {e}")
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
