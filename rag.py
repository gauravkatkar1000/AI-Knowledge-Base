"""
RAG pipeline.
Embeds query → retrieves from ChromaDB → builds prompt → calls Groq LLM.
All heavy objects are lazily initialised once and reused.
"""

import os
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "./chroma_store"
COLLECTION  = "knowledge_base"
TOP_K       = 5
LLM_MODEL   = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are an expert AI research assistant. "
    "Answer questions using ONLY the provided context. "
    "If the answer is not in the context, say so clearly. "
    "Always cite which source your answer comes from. "
    "Be concise and precise."
)

_embed_model: SentenceTransformer | None = None
_collection = None
_groq: Groq | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        _collection = client.get_collection(COLLECTION)
    return _collection


def _get_groq() -> Groq:
    global _groq
    if _groq is None:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError("GROQ_API_KEY is not set.")
        _groq = Groq(api_key=key)
    return _groq


def retrieve(query: str) -> list[dict]:
    """Embed query and return top-K chunks with metadata."""
    model = _get_embed_model()
    q_emb = model.encode([query]).tolist()[0]

    results = _get_collection().query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text":        doc,
                "source":      meta["source"],
                "title":       meta["title"],
                "chunk_index": meta["chunk_index"],
                "relevance":   max(0.0, round(1.0 - dist, 3)),
            }
        )
    return chunks


def _build_context(chunks: list[dict]) -> str:
    parts = [
        f"[Source {i}: {c['title']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    ]
    return "\n\n---\n\n".join(parts)


def answer(query: str, chat_history: list[dict] | None = None) -> dict:
    """
    Full RAG pipeline.
    Returns {"answer": str, "sources": list[dict], "error": str | None}
    """
    try:
        chunks  = retrieve(query)
        context = _build_context(chunks)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if chat_history:
            messages.extend(chat_history[-8:])

        messages.append(
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        )

        response = _get_groq().chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )

        return {"answer": response.choices[0].message.content, "sources": chunks, "error": None}

    except ValueError as exc:
        return {"answer": f"⚠️ Configuration error: {exc}", "sources": [], "error": str(exc)}
    except Exception as exc:
        return {
            "answer": "Sorry, I ran into an error fetching the response. Please try again in a moment.",
            "sources": [],
            "error":  str(exc),
        }
