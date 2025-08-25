import fitz  # PyMuPDF
from fastapi import UploadFile
import requests
import os
import logging
import tempfile


OCR_TOKEN = os.getenv("OCR_TOKEN", "")
OCR_URL = "https://ocr.fybre.me/ocrazure/?first_page_only=false"
USE_LOCAL_OCR = os.getenv("USE_LOCAL_OCR", "true").lower() in ("true", "1")


def read_upload_file(file: UploadFile, use_local_ocr: bool = USE_LOCAL_OCR) -> str:
    """
    Read and parse a FastAPI UploadFile.
    - TXT files: read directly
    - PDF files: extract using fitz (local OCR) or OCR API
    """
    if file.content_type == "text/plain":
        logging.debug(f"Reading TXT file: {file.filename}")
        return file.file.read().decode("utf-8").strip()

    if use_local_ocr:
        # Save temporarily for fitz to read
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(file.file.read())
            tmp_path = tmp_file.name

        try:
            logging.info(f"Extracting text locally with fitz: {file.filename}")
            return extract_text(tmp_path, use_local_ocr=True)
        finally:
            try:
                os.remove(tmp_path)
            except Exception as e:
                logging.warning(f"Failed to remove temp file {tmp_path}: {e}")
    else:
        # OCR API call
        try:
            logging.info(f"Sending {file.filename} to OCR API")
            response = requests.post(
                OCR_URL,
                headers={"X-Token": OCR_TOKEN},
                files={"file": (file.filename, file.file, file.content_type)},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data.get("content", "").strip()
        except Exception as e:
            logging.error(f"OCR API request failed for {file.filename}: {e}")
            return ""


def extract_text(file_path: str, use_local_ocr: bool = USE_LOCAL_OCR) -> str:
    """
    Extract text from PDF.
    - Primary: fitz (if enabled)
    - Fallback: OCR API
    """
    if use_local_ocr:
        try:
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    page_text = page.get_text()
                    if page_text:
                        text += page_text + "\n\n--- PAGE BREAK ---\n\n"

            if text.strip():
                logging.info(f"Extracted text locally from {file_path}")
                return text.strip()
            else:
                logging.debug(f"No text extracted with fitz from {file_path}")
        except Exception as e:
            logging.warning(f"fitz extraction failed for {file_path}: {e}")

    # OCR fallback
    try:
        logging.debug(f"Sending {file_path} to OCR API (fallback)")
        with open(file_path, "rb") as f:
            response = requests.post(
                OCR_URL,
                headers={"X-Token": OCR_TOKEN},
                files={"file": (os.path.basename(file_path), f, "application/pdf")},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data.get("content", "").strip()
    except Exception as e:
        logging.error(f"OCR failed for {file_path}: {e}")

    return ""
