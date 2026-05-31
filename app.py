import os
import streamlit as st

# ── Pull GROQ_API_KEY from Streamlit Cloud secrets if present ─────────────────
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from rag import answer  # noqa: E402  (must come after env setup)


# ── Auto-ingest on first start (runs once per container, cached in memory) ────
@st.cache_resource(show_spinner="🔧 Building knowledge base — scraping & embedding 7 articles (~2 min first time)…")
def _init_knowledge_base():
    from ingest import ensure_collection
    ensure_collection()
    return True

_init_knowledge_base()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Knowledge Base — Gaurav Katkar",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        #MainMenu, footer, header {visibility: hidden;}

        /* Source expander content */
        .source-item {
            border-left: 2px solid #22d3ee40;
            padding-left: 12px;
            margin-bottom: 12px;
        }
        .badge {
            display: inline-block;
            background: #22d3ee15;
            color: #22d3ee;
            border: 1px solid #22d3ee30;
            border-radius: 12px;
            padding: 1px 8px;
            font-size: 11px;
            font-family: monospace;
            margin-left: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 AI Knowledge Base")
    st.divider()

    st.markdown("### About This Demo")
    st.markdown(
        """
This is a **production-grade RAG pipeline** that answers questions using
retrieved context from 7 AI/ML Wikipedia articles — no hallucination, every
answer cites its source.

**How it works:**
1. Your question is embedded locally with Sentence Transformers
2. Top 5 relevant chunks are retrieved from ChromaDB
3. Context + question is sent to Groq's LLaMA 3.3 70B
4. Answer cites the exact source articles used
        """
    )
    st.divider()

    st.markdown("### Tech Stack")
    st.markdown(
        """
| Layer | Tool |
|---|---|
| LLM | Groq · LLaMA 3.3 70B |
| Vector DB | ChromaDB (local) |
| Embeddings | all-MiniLM-L6-v2 |
| UI | Streamlit |
        """
    )
    st.divider()

    st.markdown("### Knowledge Base")
    for article in [
        "Retrieval-Augmented Generation",
        "Large Language Models",
        "Prompt Engineering",
        "Vector Databases",
        "Transformer Architecture",
        "Anthropic",
        "Multi-Agent Systems",
    ]:
        st.markdown(f"- {article}")
    st.divider()

    st.markdown(
        "[← Back to Portfolio](https://gauravkatkar1000.vercel.app)",
        unsafe_allow_html=False,
    )
    st.markdown("Built by **Gaurav Katkar**")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 AI Knowledge Base")
st.markdown(
    "Ask questions about AI, LLMs, RAG, transformers, and more. "
    "Every answer is grounded in retrieved context — no hallucinations."
)

# ── Sample questions (shown only on empty chat) ───────────────────────────────
SAMPLES = [
    "What is retrieval augmented generation?",
    "How do vector databases work?",
    "What is prompt engineering?",
    "What makes a good LLM?",
]

if not st.session_state.messages:
    st.markdown("**Try a sample question:**")
    col1, col2 = st.columns(2)
    for i, q in enumerate(SAMPLES):
        col = col1 if i % 2 == 0 else col2
        if col.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"📚 Sources used — {len(sources)} chunks retrieved"):
        for j, src in enumerate(sources, 1):
            pct = int(src["relevance"] * 100)
            st.markdown(
                f"**{j}. {src['title']}**"
                f"<span class='badge'>{pct}% relevant</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='source-item'>"
                f"<small>{src['text'][:300]}{'…' if len(src['text']) > 300 else ''}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"[Read full article →]({src['source']})")
            if j < len(sources):
                st.divider()


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# ── Handle input ──────────────────────────────────────────────────────────────
query = st.session_state.pending
st.session_state.pending = None

user_input = st.chat_input("Ask about AI, LLMs, RAG, vector databases, transformers…")
if user_input:
    query = user_input

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            result = answer(query, chat_history=history)

        st.markdown(result["answer"])

        if result.get("sources"):
            render_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role":    "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        }
    )
