"""
One-time ingestion script.
Scrapes 7 Wikipedia articles, chunks them, embeds with sentence-transformers,
and stores in a lightweight numpy vector store. Safe to run multiple times (upsert).
"""

import sys
import requests
from bs4 import BeautifulSoup
import tiktoken
from sentence_transformers import SentenceTransformer
from vector_store import VectorStore

ARTICLES = [
    {"url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",      "title": "Retrieval-Augmented Generation"},
    {"url": "https://en.wikipedia.org/wiki/Large_language_model",                "title": "Large Language Model"},
    {"url": "https://en.wikipedia.org/wiki/Prompt_engineering",                  "title": "Prompt Engineering"},
    {"url": "https://en.wikipedia.org/wiki/Vector_database",                     "title": "Vector Database"},
    {"url": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)", "title": "Transformer Architecture"},
    {"url": "https://en.wikipedia.org/wiki/Anthropic",                           "title": "Anthropic"},
    {"url": "https://en.wikipedia.org/wiki/Multi-agent_system",                  "title": "Multi-Agent System"},
]

CHUNK_SIZE  = 300  # tokens
OVERLAP     = 50   # tokens


def scrape(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": "RAG-Demo/1.0"}, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        return ""
    for tag in content.find_all(["table", "sup", "script", "style", "figure", "figcaption", ".mw-editsection"]):
        tag.decompose()
    paras = [p.get_text().strip() for p in content.find_all("p") if p.get_text().strip()]
    return "\n\n".join(paras)


def chunk(text: str) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += CHUNK_SIZE - OVERLAP
    return chunks


def ingest():
    print("Loading embedding model (all-MiniLM-L6-v2)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    store = VectorStore()

    for article in ARTICLES:
        print(f"\n→ {article['title']}")
        try:
            text = scrape(article["url"])
            if not text:
                print("  ⚠ No text scraped, skipping.")
                continue

            chunks = chunk(text)
            print(f"  {len(chunks)} chunks")

            embeddings = model.encode(chunks, show_progress_bar=False).tolist()
            ids        = [f"{article['url']}::chunk_{i}" for i in range(len(chunks))]
            metadatas  = [
                {"source": article["url"], "title": article["title"], "chunk_index": i}
                for i in range(len(chunks))
            ]

            store.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
            print(f"  ✓ upserted")

        except Exception as exc:
            print(f"  ✗ Error: {exc}", file=sys.stderr)

    print(f"\n✓ Done. Total chunks: {store.count()}")


def ensure_collection():
    """Idempotent — only ingests if the store is empty."""
    store = VectorStore()
    if store.count() > 0:
        print(f"✓ Store ready ({store.count()} chunks). Skipping ingestion.")
        return
    print("Store empty — running ingestion…")
    ingest()


if __name__ == "__main__":
    ingest()
