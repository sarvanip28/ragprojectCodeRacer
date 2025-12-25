#!/usr/bin/env python3
"""
ingest_codes.py

Final ingestion script for ragproject.

Features:
- Loads files from a directory (pdf, docx, txt, md)
- Splits into chunks with overlap
- Validates/normalizes metadata with JSON Schema
- Embeds using OpenAI embeddings (via LangChain wrapper)
- Upserts into Chroma vector store (configurable)
- Optionally writes chunked JSONL for inspection
- CLI friendly

Dependencies (suggested):
- langchain
- chromadb
- python-magic (optional)
- pypdf, python-docx
- openai
- jsonschema
- tqdm
"""

import os
import sys
import hashlib
import json
import logging
import argparse
from typing import Dict, List, Optional, Iterable
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# third-party imports (install these)
try:
    from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.schema import Document
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    print("Missing dependencies. Install langchain, chromadb, openai, jsonschema, python-docx, pypdf.")
    raise

import jsonschema
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------
VECTOR_STORE_TYPE = os.environ.get("RAG_VECTOR_STORE", "chroma")  # "chroma" | "faiss" | "pinecone" (factory stubs)
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./.chromadb")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")  # adjust to your OpenAI model
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
JSONL_DUMP_PATH = os.environ.get("JSONL_DUMP_PATH", "./chunked_records.jsonl")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # langchain/OpenAI will read this env var by default

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ingest")

# ----------------------------
# Metadata JSON Schema
# ----------------------------
METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "source_id": {"type": "string"},
        "source_path": {"type": "string"},
        "file_name": {"type": "string"},
        "file_size": {"type": "integer"},
        "mime_type": {"type": "string"},
        "sha256": {"type": "string"},
        "ingested_at": {"type": "string", "format": "date-time"},
        "authors": {
            "type": "array",
            "items": {"type": "string"}
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["source_id", "source_path", "file_name", "file_size", "sha256", "ingested_at"]
}

# ----------------------------
# Utilities
# ----------------------------
def compute_sha256(path: Path, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()

def make_source_id(path: Path, sha256: str) -> str:
    # stable uniquely-identifying string for the file
    # safe to use as a collection/document ID prefix
    name = path.stem
    short_hash = sha256[:12]
    return f"{name}_{short_hash}"

def validate_metadata(meta: Dict) -> Dict:
    # fill defaults, then validate
    if "ingested_at" not in meta:
        meta["ingested_at"] = datetime.utcnow().isoformat() + "Z"
    try:
        jsonschema.validate(instance=meta, schema=METADATA_SCHEMA)
    except jsonschema.ValidationError as e:
        logger.error("Metadata validation error: %s", e)
        raise
    return meta

# ----------------------------
# Vector store factory
# ----------------------------
def get_vector_store(client_name: str = "chroma", collection_name: str = "rag_docs", embeddings=None):
    """
    Return a vector store instance with an 'upsert' compatible API.
    This factory uses chroma by default. Swap in FAISS or Pinecone as needed.
    """
    if client_name == "chroma":
        # persistent Chroma collection
        chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PERSIST_DIR))
        coll = chroma_client.get_or_create_collection(name=collection_name, metadata={"created_at": datetime.utcnow().isoformat()})
        # wrapper to expose a consistent upsert interface
        class ChromaWrapper:
            def __init__(self, collection, embeddings):
                self.collection = collection
                self.embeddings = embeddings

            def upsert(self, ids: List[str], embeddings_list: List[List[float]], metadatas: List[dict], documents: List[str]):
                # chroma.create / upsert
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    documents=documents
                )
                # persist if using local chroma backend
                try:
                    chroma_client.persist()
                except Exception:
                    pass
        return ChromaWrapper(coll, embeddings)
    else:
        raise NotImplementedError(f"Vector store factory not implemented for '{client_name}'")

# ----------------------------
# Loader selection
# ----------------------------
def load_file_to_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in [".txt", ".md"]:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
    elif suf in [".pdf"]:
        loader = PyPDFLoader(str(path))
        docs = loader.load()
    elif suf in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(str(path))
        docs = loader.load()
    else:
        # fallback: try to read as text
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs = [Document(page_content=text, metadata={})]
        except Exception:
            raise ValueError(f"Unsupported file type: {suf} for {path}")
    # join pages into a single string
    combined = "\n\n".join([d.page_content for d in docs if d.page_content and d.page_content.strip()])
    return combined

# ----------------------------
# Chunking
# ----------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# ----------------------------
# Main ingestion logic
# ----------------------------
@dataclass
class ChunkRecord:
    id: str
    text: str
    metadata: dict

def create_chunk_records_from_file(path: Path, extra_meta: Optional[dict] = None) -> List[ChunkRecord]:
    logger.info("Loading file: %s", path)
    text = load_file_to_text(path)
    sha = compute_sha256(path)
    source_id = make_source_id(path, sha)

    base_meta = {
        "source_id": source_id,
        "source_path": str(path.resolve()),
        "file_name": path.name,
        "file_size": path.stat().st_size,
        "mime_type": path.suffix.lower(),
        "sha256": sha,
    }
    if extra_meta:
        base_meta.update(extra_meta)

    meta_valid = validate_metadata(base_meta)

    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    records: List[ChunkRecord] = []
    for i, chunk in enumerate(chunks):
        cid = f"{meta_valid['source_id']}_chunk_{i:04d}"
        chunk_meta = dict(meta_valid)
        # per-chunk metadata helps retrieval debugging
        chunk_meta.update({
            "chunk_index": i,
            "chunk_size": len(chunk),
        })
        records.append(ChunkRecord(id=cid, text=chunk, metadata=chunk_meta))
    logger.info("Created %d chunks for %s", len(records), path.name)
    return records

def embed_and_upsert_records(records: Iterable[ChunkRecord], vector_store, embeddings):
    ids = []
    docs = []
    metadatas = []
    # compute embeddings in batches to avoid rate throttles
    batch = []
    BATCH_SIZE = 64
    from langchain.embeddings.openai import OpenAIEmbeddings as _OpenAIEmbeddings  # fallback class hint

    embed_model = embeddings
    # Some LangChain embedding wrappers accept a list and return list of vectors.
    texts_batch = []
    id_batch = []
    meta_batch = []
    for rec in records:
        texts_batch.append(rec.text)
        id_batch.append(rec.id)
        meta_batch.append(rec.metadata)
        if len(texts_batch) >= BATCH_SIZE:
            vecs = embed_model.embed_documents(texts_batch)
            vector_store.upsert(ids=id_batch, embeddings_list=vecs, metadatas=meta_batch, documents=texts_batch)
            texts_batch = []
            id_batch = []
            meta_batch = []
    # last batch
    if texts_batch:
        vecs = embed_model.embed_documents(texts_batch)
        vector_store.upsert(ids=id_batch, embeddings_list=vecs, metadatas=meta_batch, documents=texts_batch)

# ----------------------------
# Top-level helpers
# ----------------------------
def ingest_folder(folder: str,
                  vector_store_name: str = "chroma",
                  collection_name: str = "rag_docs",
                  file_glob: str = "**/*.*",
                  dry_run: bool = False,
                  dump_jsonl: bool = False,
                  extra_meta: Optional[dict] = None):
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # instantiate embeddings
    logger.info("Initializing embeddings model: %s", EMBEDDING_MODEL_NAME)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    # vector store wrapper
    vstore = get_vector_store(client_name=vector_store_name, collection_name=collection_name, embeddings=embeddings)

    # collect records across files
    all_records: List[ChunkRecord] = []
    supported_extensions = {".pdf", ".docx", ".doc", ".txt", ".md"}
    files = list(p.glob(file_glob))
    files = [f for f in files if f.suffix.lower() in supported_extensions and f.is_file()]

    if not files:
        logger.warning("No supported files found in %s (extensions: %s)", folder, supported_extensions)

    for f in tqdm(files, desc="Files"):
        try:
            recs = create_chunk_records_from_file(f, extra_meta=extra_meta)
            all_records.extend(recs)
        except Exception as e:
            logger.exception("Failed to process %s: %s", f, e)

    logger.info("Total chunks to process: %d", len(all_records))

    if dump_jsonl:
        logger.info("Dumping chunked records to %s", JSONL_DUMP_PATH)
        with open(JSONL_DUMP_PATH, "w", encoding="utf-8") as out:
            for r in all_records:
                json.dump({"id": r.id, "text": r.text, "metadata": r.metadata}, out)
                out.write("\n")

    if dry_run:
        logger.info("Dry run enabled; skipping embeddings & upsert.")
        return len(all_records)

    # do embeddings + upsert
    logger.info("Embedding and upserting into vector store (%s)...", vector_store_name)
    embed_and_upsert_records(all_records, vstore, embeddings)
    logger.info("Ingestion complete. %d chunks upserted.", len(all_records))
    return len(all_records)

# ----------------------------
# CLI
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser("ingest_codes")
    p.add_argument("--folder", "-f", required=True, help="Folder containing files to ingest")
    p.add_argument("--collection", "-c", default="rag_docs", help="Vector store collection name")
    p.add_argument("--vector-store", "-v", default=VECTOR_STORE_TYPE, help="Vector store (chroma/faiss/pinecone)")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size (chars)")
    p.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Chunk overlap (chars)")
    p.add_argument("--dry-run", action="store_true", help="Only chunk and validate metadata; do not embed/upsert")
    p.add_argument("--dump-jsonl", action="store_true", help="Dump chunked JSONL to --jsonl-path")
    p.add_argument("--jsonl-path", default=JSONL_DUMP_PATH, help="Where to dump chunked JSONL")
    p.add_argument("--extra-meta", default=None, help="JSON string of extra metadata to add to each document")
    return p

def main_cli(argv=None):
    global CHUNK_SIZE, CHUNK_OVERLAP, JSONL_DUMP_PATH
    args = build_parser().parse_args(argv)
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    JSONL_DUMP_PATH = args.jsonl_path
    extra_meta = json.loads(args.extra_meta) if args.extra_meta else None

    count = ingest_folder(
        folder=args.folder,
        vector_store_name=args.vector_store,
        collection_name=args.collection,
        file_glob="**/*.*",
        dry_run=args.dry_run,
        dump_jsonl=args.dump_jsonl,
        extra_meta=extra_meta
    )
    logger.info("Done. Processed %d chunks.", count)

if __name__ == "__main__":
    main_cli()
