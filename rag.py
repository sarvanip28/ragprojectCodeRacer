# app/rag.py
"""
RAG helper using sentence-transformers + chromadb (lazy, defensive).
This variant constructs chromadb.Client() in the simple, modern way to avoid legacy-settings errors.

Provides:
- add_documents(doc_texts, metadatas=None, ids=None)
- retrieve_similar(text, k=5) -> List[str]
- is_rag_available()
"""

from __future__ import annotations
import os
import logging
from typing import List, Optional, Dict

LOG = logging.getLogger("coderacer.rag")
LOG.setLevel(os.getenv("LOG_LEVEL", "INFO"))

_DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
_INDEX_NAME = os.getenv("CHROMA_INDEX_NAME", "coderacer_index")

_client = None
_collection = None
_embed_model = None


def _ensure_init():
    """Lazy init chromadb client, collection and sentence-transformers model."""
    global _client, _collection, _embed_model
    if _client is not None and _collection is not None and _embed_model is not None:
        return

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        LOG.debug("RAG libs missing: %s", e)
        raise RuntimeError("Missing chromadb or sentence-transformers") from e

    # load embedding model
    _embed_model = SentenceTransformer(_DEFAULT_EMBED_MODEL)

    # Create client using the simple, modern constructor.
    # This avoids legacy Settings error on some chromadb installs.
    try:
        # Use default client which will pick a working persistence backend.
        _client = chromadb.Client()
    except Exception as e:
        # Last-resort: try a minimal Settings config that does not trigger legacy path
        try:
            from chromadb.config import Settings
            settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=_PERSIST_DIR)
            _client = chromadb.Client(settings=settings)
        except Exception as e2:
            LOG.exception("Failed to create chromadb client: %s ; fallback also failed: %s", e, e2)
            raise

    # get or create collection (newer chroma exposes get_or_create_collection)
    try:
        # preferred: get_or_create_collection if available
        if hasattr(_client, "get_or_create_collection"):
            _collection = _client.get_or_create_collection(name=_INDEX_NAME)
        else:
            # fallback to get_collection / create_collection
            try:
                _collection = _client.get_collection(name=_INDEX_NAME)
            except Exception:
                _collection = _client.create_collection(name=_INDEX_NAME)
    except Exception:
        # If collection operations fail, log and re-raise to allow caller to handle
        LOG.exception("Failed to get or create chroma collection")
        raise

    LOG.info("RAG initialized: index=%s persist=%s model=%s", _INDEX_NAME, _PERSIST_DIR, _DEFAULT_EMBED_MODEL)


def add_documents(doc_texts: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
    """
    Add documents to the RAG collection.
    - doc_texts: list of strings
    - metadatas: optional list of dicts (same length)
    - ids: optional list of ids
    """
    try:
        _ensure_init()
    except RuntimeError:
        raise RuntimeError("RAG dependencies not available; cannot add documents")

    if metadatas and len(metadatas) != len(doc_texts):
        raise ValueError("metadatas length must match doc_texts")
    if ids and len(ids) != len(doc_texts):
        raise ValueError("ids length must match doc_texts")

    emb = _embed_model.encode(doc_texts, show_progress_bar=False, convert_to_numpy=True)

    if ids is None:
        ids = [f"doc_{i}" for i in range(len(doc_texts))]

    embeddings = [e.tolist() if hasattr(e, "tolist") else e for e in emb]

    _collection.add(
        documents=doc_texts,
        metadatas=metadatas or [{} for _ in doc_texts],
        ids=ids,
        embeddings=embeddings,
    )

    try:
        # Persist if client supports it
        if hasattr(_client, "persist"):
            _client.persist()
    except Exception:
        LOG.debug("Chroma persist not required or failed (ignored).")


def retrieve_similar(text: str, k: int = 5) -> List[str]:
    """Return up to k similar documents (strings). If RAG not available or index empty, returns []."""
    if not text:
        return []

    try:
        _ensure_init()
    except RuntimeError:
        LOG.debug("RAG not available; retrieve_similar returns empty list.")
        return []

    try:
        q_emb = _embed_model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
    except Exception as e:
        LOG.exception("Failed to embed query: %s", e)
        return []

    try:
        res = _collection.query(
            embeddings=[q_emb.tolist() if hasattr(q_emb, "tolist") else q_emb],
            n_results=k,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        LOG.exception("Chroma query failed: %s", e)
        return []

    docs = []
    docs_list = res.get("documents", [[]])[0] if "documents" in res else []
    metadatas_list = res.get("metadatas", [[]])[0] if "metadatas" in res else [None] * len(docs_list)

    for doc, meta in zip(docs_list, metadatas_list):
        if meta and isinstance(meta, dict) and meta.get("text"):
            docs.append(meta.get("text"))
        else:
            docs.append(doc)
    return docs


def is_rag_available() -> bool:
    try:
        import chromadb, sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False
