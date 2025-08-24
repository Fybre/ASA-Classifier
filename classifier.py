from typing import Dict
import pandas as pd
from llama_index.core import VectorStoreIndex

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from chromadb.config import Settings

from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

import logging
from openai import AzureOpenAI
import ollama
import chromadb
import os
import json
import re

# ---------------------------
# Config
# ---------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
DB_PATH = os.getenv("DB_PATH", "db/chroma_asa_rrds")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "Not Configured")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1")
LLM_KEY = os.getenv("LLM_KEY", "")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "")

TOP_K_CANDIDATES = 5
SIMILARITY_THRESHOLD = 0.30  # adjust based on accuracy/performance tradeoff

# ---------------------------
# Logger Setup
# ---------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# ---------------------------
# Utility
# ---------------------------
def safe_get(row, key):
    """Safely get a value from a DataFrame row as a stripped string."""
    return str(row.get(key, "")).strip()


# ---------------------------
# ASA Definitions Loader
# ---------------------------
def get_asa_definitions(csv_file):
    """
    Load ASA codes and definitions from a CSV file into a single text block.
    Used for prompting the LLM.
    """
    logger.info(f"Loading ASA definitions from {csv_file}...")
    df = pd.read_csv(csv_file, dtype=str, keep_default_na=False)

    asa_definitions = "The following are the ASA codes and their definitions:\n\n"
    for idx, row in df.iterrows():
        asa_definitions += (
            f"{safe_get(row,'Code')} (Classification Hierarchy: "
            f"{safe_get(row,'Function')} / {safe_get(row,'Class')} / "
            f"{safe_get(row,'SubClass')} / {safe_get(row,'Detail')}, "
            f"Description: {safe_get(row,'Description')}, "
            f"Example Documents: {safe_get(row,'Example Document Types')})\n\n"
        )
    logger.info(f"Loaded {len(df)} ASA definitions.")
    return asa_definitions


# ---------------------------
# LLM Backends
# ---------------------------
def call_llm(prompt: str) -> str:
    """
    Call the configured LLM provider (Ollama, Azure, or OpenAI) with the given prompt.
    Returns the raw text response.
    """
    logger.debug(f"Calling LLM provider '{LLM_PROVIDER}' with model '{LLM_MODEL}'")

    if LLM_PROVIDER == "ollama":
        options={"num_predict": 512, "top_p": 0.9, "top_k": 40, "temperature": 0.3}
        response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], options=options)
        return response["message"]["content"]

    elif LLM_PROVIDER == "azure":
        client = AzureOpenAI(
            api_key=LLM_KEY,
            api_version="2025-01-01-preview",
            azure_endpoint=LLM_ENDPOINT
        )
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content

    elif LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=LLM_KEY)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


# ---------------------------
# Main Classification Orchestrator
# ---------------------------
def classify_document(text: str, use_similarity: bool = True, use_llm: bool = True,
                      top_k_candidates=TOP_K_CANDIDATES, similarity_threshold=SIMILARITY_THRESHOLD,
                      source_filter=None):
    """
    Classify a document using:
    - LLM classification (based on ASA definitions)
    - Similarity search (based on vector embeddings + Chroma index)
    Combines results and attempts a "voted" choice.
    """
    logger.info("Classifying document...")
    llm_result = classify_document_llm_using_asa(text) if use_llm else None
    similarity_result = classify_document_similarity(text, source_filter=source_filter) if use_similarity else None

    asa_code_obj = llm_result.get("asa_llm_choice", {}) if llm_result else {}
    asa_code = asa_code_obj.get("code")

    similarity_code = similarity_result.get("similarity_llm_choice", {}).get("code") if similarity_result else None

    # Voting logic: only confirm if both match
    voted_choice = (
        asa_code if use_llm and use_similarity and asa_code == similarity_code
        else asa_code if use_llm and not use_similarity
        else similarity_code if use_similarity and not use_llm
        else None
    )

    logger.info(f"LLM choice: {asa_code}, Similarity choice: {similarity_code}, Voted choice: {voted_choice}")

    return {
        "asa_llm_choice": llm_result,
        "similarity_choice": similarity_result,
        "voted_choice": voted_choice
    }


# ---------------------------
# LLM-only ASA Classification
# ---------------------------
def classify_document_llm_using_asa(text: str) -> Dict:
    """
    Classify a document using the LLM only.
    It compares the document text to the full set of ASA definitions.
    """
    logger.debug("Classifying using LLM with ASA definitions...")
    prompt = f"""
    You are a document classifier.

Document:
\"\"\"{text}\"\"\"

ASA Definitions:
\"\"\"{asa_definitions}\"\"\"

Task: Classify the document into a single ASA code based on the definitions provided.
Choose the most relevant code that fits the document content, giving preference to STUDENT MANAGEMENT codes when multiple are relevant.

Return your answer as strictly valid JSON only, with this exact format (no extra text before or after):

{{
  "code": "<ASA code>", 
  "Explanation": "<a short explanation why this code fits>"
}}

Notes:
- The "code" field must be ONLY the numerical ASA code, e.g. "5.1.2" or "4.1" — no extra words or characters.
- The code consists only of digits and dots as separators.
- Only output the JSON object. No additional commentary, text, or formatting.

Example:

{{
  "code": "5.1.2", 
  "Explanation": "The document discusses attendance policies relevant to student management under code 5.1.2."
}}
"""

    llm_answer = call_llm(prompt=prompt).strip()
    logger.debug(f"Raw LLM response: {llm_answer[:200]}...")

    # Extract JSON
    json_match = re.search(r"\{.*\}", llm_answer, re.DOTALL)
    if json_match:
        llm_answer = json_match.group(0)

    try:
        llm_result = json.loads(llm_answer)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON")
        llm_result = {"code": None, "Explanation": "Unable to parse LLM response"}

    return {"asa_llm_choice": llm_result}


# ---------------------------
# Similarity-based Classification
# ---------------------------
def classify_document_similarity(text: str,
                                 top_k_candidates=TOP_K_CANDIDATES,
                                 similarity_threshold=SIMILARITY_THRESHOLD,
                                 source_filter=None) -> Dict:
    """
    Classify a document using semantic similarity search from the Chroma vector store.
    Optionally filters results by metadata (e.g., source).
    """
    logger.debug("Classifying using similarity search...")
    metadata_filter = None
    if source_filter:
        metadata_filter = MetadataFilters(filters=[
            MetadataFilter(key="source", value=source_filter, operator=FilterOperator.EQ)
        ])

    retriever = index.as_retriever(
        similarity_top_k=top_k_candidates,
        filters=metadata_filter if metadata_filter else None
    )

    results = retriever.retrieve(text)

    candidates = []
    for r in results:
        if not r.node.metadata or r.score is None:
            logger.warning(f"Skipping result with no metadata: {r.node.id}")
            continue
        candidates.append({
            "score": float(r.score),
            "code": r.node.metadata.get("code", "Unknown"),
            "function": r.node.metadata.get("function", ""),
            "class": r.node.metadata.get("class", ""),
            "subclass": r.node.metadata.get("subclass", ""),
            "detail": r.node.metadata.get("detail", ""),
            "description": r.node.metadata.get("description", ""),
            "example_document_types": r.node.metadata.get("example_document_types", ""),
            "category_mapping": r.node.metadata.get("category_mapping", ""),
            "source": r.node.metadata.get("source", "unknown")
        })

    logger.debug(f"Retrieved {len(candidates)} candidates")

    # Filter by threshold
    filtered_candidates = [c for c in candidates if c["score"] >= similarity_threshold]

    # Fallback to top candidate if nothing passes threshold
    fallback_used = False
    if not filtered_candidates and candidates:
        filtered_candidates = [max(candidates, key=lambda x: x["score"])]
        fallback_used = True
        logger.info("No candidates passed threshold — using fallback top candidate")

    limited_candidates = filtered_candidates[:top_k_candidates]

    candidate_texts = "\n".join([
        f"- {c['code']} ({c['function']} / {c['class']} / {c['subclass']} / {c['detail']} / "
        f"{c['description']} / {c['example_document_types']} / {c['category_mapping']}) — score: {c['score']:.3f}"
        for c in limited_candidates
    ])

    # Build prompt for LLM selection
    prompt = f"""You are a document classifier.

Document:
\"\"\"{text}\"\"\"

Candidate codes with similarity scores:
{candidate_texts}

Task: Pick the single best code from the list above.

Return your answer as strictly valid JSON only, with this exact format (no extra text before or after):
{{
  "code": "<ASA code>", 
  "Explanation": "<a short explanation why this code fits>"
}}

Notes:
- The "code" field must be ONLY the numerical ASA code, e.g. "5.1.2" or "4.1" — no extra words or characters.
- The code consists only of digits and dots as separators.
- Only output the JSON object. No additional commentary, text, or formatting.

Example:

{{
  "code": "5.1.2", 
  "Explanation": "The document discusses attendance policies relevant to student management under code 5.1.2."
}}
"""


    llm_answer = call_llm(prompt=prompt).strip()
    logger.debug(f"Raw LLM response (similarity): {llm_answer[:200]}...")

    json_match = re.search(r"\{.*\}", llm_answer, re.DOTALL)
    if json_match:
        llm_answer = json_match.group(0)

    try:
        llm_result = json.loads(llm_answer)
    except json.JSONDecodeError:
        logger.error("Failed to parse similarity LLM response as JSON")
        llm_result = {
            "code": limited_candidates[0]["code"] if limited_candidates else None,
            "Explanation": "Fallback: No similar candidates found" if limited_candidates else "Fallback: Could not parse LLM response"
        }

    return {
        "similarity_retrieved_candidates": candidates,
        "similarity_llm_choice": llm_result,
        "similarity_fallback_used": fallback_used
    }


# ---------------------------
# Index Loader
# ---------------------------
def load_index(path, model=EMBED_MODEL) -> VectorStoreIndex:
    """
    Load a persistent Chroma vector store index with HuggingFace embeddings.
    Raises FileNotFoundError if DB_PATH does not exist.
    """
    if os.path.exists(DB_PATH):
        logger.info(f"Loading index from {path} with model '{model}'")
        chroma_client = chromadb.PersistentClient(path, settings=Settings(anonymized_telemetry=False))
        chroma_collection = chroma_client.get_collection("asa_rrds")
        embed_model = HuggingFaceEmbedding(model_name=model)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)
    else:
        logger.error(f"Index not found at {path}. Please run the embedder first.")
        raise FileNotFoundError(f"Index not found at {path}")


# ---------------------------
# Init Globals
# ---------------------------
asa_definitions = get_asa_definitions("data/asa_rrds.csv")
index = load_index(DB_PATH, EMBED_MODEL)
