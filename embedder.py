import json
import pandas as pd
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.core.storage import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from chromadb.config import Settings

from text_extraction import extract_text

import chromadb
import os
import shutil
import logging

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
DB_PATH = "db/chroma_asa_rrds"

# ---------------------------
# Logger Setup
# ---------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class ASACode:
    def __init__(self, code, function, class_name, subclass=None, detail=None, description=None, example_document_types=None, category_mapping=None, source=None):
        self.code = code
        self.function = function
        self.class_name = class_name
        self.subclass = subclass
        self.detail = detail
        self.description = description
        self.example_document_types = example_document_types
        self.category_mapping = category_mapping
        self.source = source

    def to_dict(self):
        return {
            "code": self.code,
            "function": self.function,
            "class": self.class_name,
            "subclass": self.subclass,
            "detail": self.detail,
            "description": self.description,
            "example_document_types": self.example_document_types,
            "category_mapping": self.category_mapping,
            "source": self.source
        }
    def __str__(self):
        # Nicely formats all properties in a readable string
        return (
            f"Code: {self.code} ({self.function} / {self.class_name or 'N/A'} / {self.subclass or 'N/A'} / {self.detail or 'N/A'} / Description: {self.description or 'N/A'} / "
            f"Example Documents: {self.example_document_types or 'N/A'} / Source: {self.source or 'N/A'})")

def get_asa_details_from_csv(csv_file): 
    """
    Load ASA documents from a CSV file.
    """
    logger.info(f"Loading ASA documents from {csv_file}...")
    df = pd.read_csv(csv_file, dtype=str, keep_default_na=False)
    asa_docs = []
    
    for idx, row in df.iterrows():
        # Create ASACode object
        code_obj = ASACode(
            code=row["Code"].strip(),
            function=row["Function"].strip(),
            class_name=row["Class"].strip(),
            subclass=row.get("SubClass", "").strip(),
            detail=row.get("Detail", "").strip(),
            description=row.get("Description", "").strip(),
            example_document_types=row.get("Example Document Types", "").strip(),
            category_mapping=row.get("Category Mapping", "").strip(),
        )
        asa_docs.append(code_obj)
    return asa_docs

def get_asa_documents(asa_detail_list):
    """
    Convert ASA details to LlamaIndex Document objects.
    """
    logger.info("Converting ASA details to LlamaIndex Document objects...")
    documents = []
    
    for code_obj in asa_detail_list:
        # Create a Document from the ASACode object
        code_obj.source = "asa_rrds"  # Default source if not provided
        doc = Document(text=str(code_obj), metadata=code_obj.to_dict())
        documents.append(doc)
    
    return documents

def get_custom_metadata(filename):
    """
    Load metadata for a file based on its parent directory name.
    Looks up ASA details using the directory name as a code.
    """
    code = os.path.basename(os.path.dirname(filename)).strip()

    try:
        asa_detail = next((item for item in asa_details if item.code == code), None)
    except NameError:
        logger.error("asa_details is not defined.")
        return {"code": code}

    if not asa_detail:
        logger.warning(f"No ASA detail found for code '{code}' in {filename}")
        return {"code": code}

    asa_detail.source = "reference"
    logger.debug(f"Loading metadata for {filename}: {asa_detail}")

    return asa_detail.to_dict()


def get_reference_documents(directory):
    """
    Load reference documents from a directory.
    """
    logger.info(f"Loading reference documents from {directory}...")
    try:
        reader = SimpleDirectoryReader(directory, recursive=True, file_metadata=get_custom_metadata,required_exts=[".txt"])
        return reader.load_data()
    except Exception as e:
        logger.error(f"Error reading directory {directory}: {e}")
        return []


def convert_documents_to_text(directory):
    """
    Convert all documents in a directory to text format.
    """
    logger.info(f"Converting documents in {directory} to text format...")
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)

            # Skip .txt files themselves
            if not file_path.endswith(".pdf"):
                continue

            # Expected sidecar .txt file (strip extension, add .txt)
            txt_path = os.path.splitext(file_path)[0] + ".txt"

            if os.path.exists(txt_path):
                logger.info(f"✅ Skipping (already has .txt): {file_path}")
                continue

            # Extract text and save
            logger.info(f"🔍 Extracting text from: {file_path}")
            try:
                text = extract_text(file_path, use_fitz=False)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                logger.info(f"💾 Saved: {txt_path}")
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path}: {e}")
        

# Ensure the database directory is clean
if os.path.exists("db/chroma_asa_rrds"):
    logger.info("Clearing previous ChromaDB database...")
    shutil.rmtree(DB_PATH, ignore_errors=True)  # Clear previous database

# Initialize ChromaDB client and collection
logger.info("Initializing ChromaDB client...")

chroma_client = chromadb.PersistentClient(DB_PATH, settings=Settings(anonymized_telemetry=False))
chroma_collection = chroma_client.get_or_create_collection("asa_rrds")
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

# Convert all documents in the reference directory to text format
logger.info("Converting reference documents to text format...")
convert_documents_to_text("data/reference")

# Load all documents from both ASA RRDS and reference directories
logger.info("Loading all documents...")
# Load ASA RRDS details and convert to documents
asa_details = get_asa_details_from_csv("data/asa_rrds.csv")
# Convert ASA details to LlamaIndex Document objects
asa_documents = get_asa_documents(asa_detail_list=asa_details)
# Load reference documents from the specified directory
reference_documents = get_reference_documents("data/reference")



all_documents = asa_documents + reference_documents

# Set up Chroma vector store

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# Create the index from the nodes
logger.info("Creating index from documents...")
index = VectorStoreIndex.from_documents(
    all_documents, storage_context=storage_context, embed_model=embed_model
)

logger.info("All classes and descriptions stored to ChromaDB via LlamaIndex.")

