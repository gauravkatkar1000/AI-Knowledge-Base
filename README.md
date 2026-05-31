# AI Knowledge Base — RAG Chatbot Demo

A production-grade RAG chatbot that answers questions about AI/ML concepts by retrieving context from 7 curated Wikipedia articles. Built with Groq (LLaMA 3.3 70B), ChromaDB, Sentence Transformers, and Streamlit.

**Live demo:** *(add your Streamlit Cloud URL here)*  
**Portfolio:** https://portfolio-theta-dun-76.vercel.app

---

## What This Does

1. **Ingestion** — 7 Wikipedia articles are scraped, chunked into 300-token segments with 50-token overlap, embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB locally.
2. **Retrieval** — Your question is embedded and the top 5 most relevant chunks are retrieved using cosine similarity.
3. **Generation** — Retrieved chunks + question are sent to Groq's LLaMA 3.3 70B. The model answers using *only* the provided context and cites its sources.

**Knowledge base covers:** RAG, Large Language Models, Prompt Engineering, Vector Databases, Transformer Architecture, Anthropic, Multi-Agent Systems.

---

## Run Locally

### 1. Clone and install
```bash
git clone https://github.com/<your-username>/rag-demo.git
cd rag-demo
pip install -r requirements.txt
```

### 2. Set your Groq API key
```bash
cp .env.example .env
# Edit .env and add your key: GROQ_API_KEY=gsk_...
```
Get a free key at https://console.groq.com

### 3. Build the vector store (one time)
```bash
python ingest.py
```
This scrapes Wikipedia, creates embeddings, and saves to `./chroma_store/`.  
Safe to re-run — uses upsert so no duplicate data.

### 4. Start the app
```bash
streamlit run app.py
```

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (make sure `chroma_store/` is committed — it's pre-built so cloud skips re-ingestion).

2. Go to https://share.streamlit.io → **New app** → select your repo → set main file to `app.py`.

3. In **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```

4. Click **Deploy**. First boot downloads the embedding model (~90MB, cached after that).

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq API · LLaMA 3.3 70B Versatile |
| Vector DB | ChromaDB (local persistent) |
| Embeddings | Sentence Transformers · all-MiniLM-L6-v2 |
| Chunking | tiktoken · cl100k_base |
| UI | Streamlit |
| Scraping | requests + BeautifulSoup4 |

---

## Project Structure

```
rag-demo/
├── app.py            # Streamlit UI
├── rag.py            # RAG pipeline (retrieve + generate)
├── ingest.py         # One-time ingestion script
├── requirements.txt
├── .env.example      # Copy to .env and add your key
├── .gitignore
├── .streamlit/
│   └── config.toml   # Dark theme config
└── chroma_store/     # Pre-built vector DB (committed to repo)
```
