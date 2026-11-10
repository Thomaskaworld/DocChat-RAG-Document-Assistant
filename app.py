"""DocChat — polished Streamlit interface for document-grounded chat."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

import ingest as ing
import rag

DOCS_DIR = Path("docs")
st.set_page_config(
    page_title="DocChat | Your document workspace",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------- visual system
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#e9edf7; --muted:#9aa6bd; --panel:#151c2b; --line:rgba(148,163,184,.17); --accent:#8b7cff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 75% 0%, rgba(93,76,190,.16), transparent 34%), #0b1020; color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#0e1524; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
.block-container { max-width:1250px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
h1 { font-weight:800 !important; }
.brand { display:flex; align-items:center; gap:12px; margin-bottom:6px; }
.brand-icon { width:42px;height:42px;border-radius:14px;background:linear-gradient(135deg,#8b7cff,#4f8cff);display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 8px 25px rgba(110,95,255,.25); }
.brand-name { font:800 22px 'Manrope',sans-serif; color:#f4f5ff; }
.brand-sub { color:#93a0b8; font-size:12px; margin:0 0 22px 54px; }
.hero { padding:26px 28px; border:1px solid var(--line); border-radius:22px; background:linear-gradient(120deg,rgba(139,124,255,.16),rgba(79,140,255,.07) 55%,rgba(21,28,43,.75)); margin-bottom:22px; }
.eyebrow { color:#b9adff; text-transform:uppercase; font-size:11px; font-weight:700; letter-spacing:.15em; }
.hero h1 { font-size:34px; margin:7px 0 8px; color:#f7f7ff; }
.hero p { color:#aeb9ce; margin:0; max-width:700px; }
.stat-card { border:1px solid var(--line); background:rgba(21,28,43,.78); border-radius:16px; padding:15px 17px; min-height:88px; }
.stat-label { color:#94a3b8; font-size:12px; margin-bottom:5px; }
.stat-value { font:800 23px 'Manrope',sans-serif; color:#f4f5ff; }
.section-label { color:#9daac1; text-transform:uppercase; letter-spacing:.13em; font-size:11px; font-weight:700; margin:22px 0 10px; }
div[data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:18px; background:rgba(21,28,43,.72); padding:14px 17px; margin-bottom:12px; }
[data-testid="stChatInput"] { border:1px solid rgba(139,124,255,.35); border-radius:18px; background:#151c2b; }
.stButton > button { border-radius:12px; font-weight:700; border:1px solid var(--line); }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#7868ee,#5d83ed); border:0; color:white; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:13px; background:rgba(15,22,36,.55); }
[data-testid="stMetric"] { background:rgba(21,28,43,.78); padding:12px 14px; border:1px solid var(--line); border-radius:14px; }
[data-testid="stFileUploader"] { border:1px dashed rgba(139,124,255,.45); border-radius:14px; background:rgba(139,124,255,.045); }
small, .muted { color:#95a2b8; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ------------------------------- helpers
def indexed_count() -> int:
    try:
        return int(ing.get_collection().count())
    except Exception:
        return 0


def indexed_sources() -> list[str]:
    try:
        got = ing.get_collection().get(include=["metadatas"], limit=10000)
        return sorted({m.get("source", "Unknown") for m in got.get("metadatas", [])})
    except Exception:
        return []


def ingest_files(paths: list[Path], size: int, overlap: int, reset: bool) -> str:
    records: list[dict] = []
    for path in paths:
        records.extend(ing.read_pdf(path) if path.suffix.lower() == ".pdf" else ing.read_text(path))
    if not records:
        raise ValueError("No extractable text found. Scanned PDFs may need OCR first.")
    chunks = ing.chunk(records, size, overlap)
    embedder = rag._embedder()
    vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True, batch_size=64)
    collection = ing.get_collection(reset=reset)
    for start in range(0, len(chunks), 256):
        batch = chunks[start:start + 256]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
            embeddings=[v.tolist() for v in vectors[start:start + 256]],
        )
    if hasattr(rag, "_collection"):
        rag._collection.cache_clear()
    return f"Successfully indexed {len(records)} page(s) into {len(chunks)} chunks."


def backend_key_env(name: str):
    configs = getattr(rag, "BACKENDS", {})
    if name in configs:
        return configs[name].get("key_env")
    return {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "ollama": None}.get(name)


def render_sources(hits: list[dict], key: str):
    if not hits:
        return
    with st.expander(f"📚  View {len(hits)} retrieved source passages", expanded=False):
        for i, hit in enumerate(hits, 1):
            source = Path(str(hit.get("source", "Unknown"))).name
            page = hit.get("page", "?")
            score = hit.get("similarity")
            score_text = f" · similarity {score:.3f}" if isinstance(score, (int, float)) else ""
            st.markdown(f"**[{i}] {source}**  ·  Page {page}{score_text}")
            st.code(str(hit.get("text", ""))[:1800], language="text")
            if i != len(hits):
                st.divider()

# ------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">📚</div><div class="brand-name">DocChat</div></div><div class="brand-sub">Your private document intelligence workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Knowledge base</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Add PDF, TXT or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="docchat_knowledge_base_uploader",
    )
    with st.expander("⚙️  Indexing settings", expanded=False):
        c1, c2 = st.columns(2)
        chunk_size = c1.number_input(
            "Chunk size", min_value=200, max_value=3000,
            value=800, step=100, key="docchat_chunk_size"
        )
        overlap = c2.number_input(
            "Overlap", min_value=0, max_value=500,
            value=120, step=20, key="docchat_chunk_overlap"
        )
        reset_first = st.checkbox("Replace current index", value=False, key="docchat_reset_index")
    if st.button("＋  Ingest documents", type="primary", use_container_width=True, disabled=not uploads, key="docchat_ingest_button"):
        DOCS_DIR.mkdir(exist_ok=True)
        saved = []
        for upload in uploads:
            safe_name = Path(upload.name).name
            destination = DOCS_DIR / safe_name
            destination.write_bytes(upload.getbuffer())
            saved.append(destination)
        try:
            with st.spinner("Reading documents, creating embeddings and updating your knowledge base…"):
                message = ingest_files(saved, int(chunk_size), int(overlap), reset_first)
            st.success(message)
            st.rerun()
        except Exception as exc:
            st.error(f"Ingestion failed: {type(exc).__name__}: {exc}")

    st.markdown('<div class="section-label">Workspace settings</div>', unsafe_allow_html=True)
    available_sources = indexed_sources()
    selected_source = st.selectbox("Search scope", ["All documents"] + available_sources, key="docchat_search_scope")
    top_k = st.slider("Retrieved passages", 2, 10, 4, key="docchat_top_k")
    backends = ["gemini", "groq", "ollama"]
    current_backend = os.getenv("LLM_BACKEND", "gemini").lower()
    backend = st.selectbox("AI provider", backends, index=backends.index(current_backend) if current_backend in backends else 0, key="docchat_ai_provider")
    key_env = backend_key_env(backend)
    key_ready = bool(rag.get_key(key_env)) if key_env else True
    if backend == "ollama":
        st.caption("🖥️ Local Ollama provider")
    else:
        st.caption("🟢 Provider key detected" if key_ready else f"🟠 Add {key_env} to your .env file")

    st.markdown('<div class="section-label">Knowledge base status</div>', unsafe_allow_html=True)
    st.metric("Indexed chunks", indexed_count())
    st.caption(f"{len(available_sources)} document(s) indexed")
    if available_sources:
        with st.expander("View indexed documents"):
            for item in available_sources:
                st.markdown(f"📄 `{Path(item).name}`")
    if st.button("Clear knowledge base", use_container_width=True, key="docchat_clear_index"):
        if hasattr(rag, "_collection"):
            rag._collection.cache_clear()
        shutil.rmtree(getattr(ing, "DB_DIR", "chroma_db"), ignore_errors=True)
        st.session_state.messages = []
        st.rerun()

# ------------------------------- main workspace
count = indexed_count()
sources = indexed_sources()
st.markdown(
    '<div class="hero"><div class="eyebrow">AI-powered document assistant</div>'
    '<h1>Ask your documents.<br>Get answers you can trace.</h1>'
    '<p>Upload your knowledge, ask questions in natural language, and explore grounded answers with source passages and page references.</p></div>',
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="stat-card"><div class="stat-label">Indexed passages</div><div class="stat-value">{count:,}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="stat-card"><div class="stat-label">Documents in workspace</div><div class="stat-value">{len(sources)}</div></div>', unsafe_allow_html=True)
with m3:
    status = "Ready" if key_ready else "API key needed"
    st.markdown(f'<div class="stat-card"><div class="stat-label">AI provider · {backend.upper()}</div><div class="stat-value" style="font-size:20px">{status}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)
if "messages" not in st.session_state:
    st.session_state.messages = []

if count == 0 and not st.session_state.messages:
    st.info("Your workspace is ready. Add a PDF or text document from the sidebar, then ingest it to start chatting.")
    st.markdown("**Try asking:**  `Summarize this document` · `What are the key points?` · `What does it say about …?`")

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "📚"):
        st.markdown(message["content"])
        render_sources(message.get("hits", []), f"history_{index}")

if question := st.chat_input("Ask a question about your indexed documents…", key="docchat_chat_input"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📚"):
        if count == 0:
            reply, hits = "No documents are indexed yet. Add a document in the sidebar first.", []
        else:
            with st.spinner("Searching your documents and preparing an answer…"):
                try:
                    result = rag.answer(
                        question,
                        k=int(top_k),
                        source=None if selected_source == "All documents" else selected_source,
                        backend=backend,
                    )
                    reply, hits = result.get("answer", "No answer returned."), result.get("hits", [])
                except Exception as exc:
                    reply, hits = f"Something went wrong: {type(exc).__name__}: {exc}", []
        st.markdown(reply)
        render_sources(hits, "current")
    st.session_state.messages.append({"role": "assistant", "content": reply, "hits": hits})

st.markdown('<p class="muted" style="text-align:center;font-size:11px;margin-top:28px">DocChat · Answers are generated from retrieved document passages. Always verify important information against the original source.</p>', unsafe_allow_html=True)
