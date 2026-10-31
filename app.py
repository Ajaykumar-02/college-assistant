import os
import gc
import hashlib
import tempfile
import streamlit as st
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0a0f1e;
    --surface: #101729;
    --surface-2: #172036;
    --border: #26314d;
    --accent: #3b82f6;
    --accent-glow: #60a5fa;
    --accent-2: #fbbf24;
    --text: #e6edf3;
    --text-muted: #8a98b8;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(59, 130, 246, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59, 130, 246, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.card:hover {
    border-color: var(--accent);
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.badge-purple { background: rgba(59,130,246,0.2); color: var(--accent-glow); border: 1px solid rgba(59,130,246,0.3); }
.badge-cyan   { background: rgba(251,191,36,0.15); color: var(--accent-2);    border: 1px solid rgba(251,191,36,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }

/* Query-route badges */
.badge-academic { background: rgba(59,130,246,0.2); color: var(--accent-glow); border: 1px solid rgba(59,130,246,0.3); }
.badge-fee      { background: rgba(251,191,36,0.15); color: var(--accent-2);    border: 1px solid rgba(251,191,36,0.3); }
.badge-general  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }

/* ── Input & Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(59,130,246,0.4) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
}

/* ── Status rows (knowledge base) ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.4rem 0 1rem 0;
    border: 1px solid var(--border);
    font-size: 0.78rem;
    word-break: break-word;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-done     { background: var(--success); box-shadow: 0 0 8px var(--success); }
.dot-pending  { background: var(--border); }
.dot-error    { background: var(--danger); box-shadow: 0 0 8px var(--danger); }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(59,130,246,0.10) !important;
    border-color: rgba(59,130,246,0.25);
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(251,191,36,0.08) !important;
    border-color: rgba(251,191,36,0.2);
}

/* ── Chat input ── */
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background: var(--bg) !important;
}

[data-testid="stChatInput"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Stale Streamlit elements ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* Developer credit */
.dev-credit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    text-align: center;
}
.dev-credit span {
    color: var(--accent-glow);
    font-weight: 600;
}
.dev-footer {
    margin-top: 3rem;
    margin-bottom: 5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.dev-footer span {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-glow), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Social / contact links */
.dev-links {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}
.dev-links a {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.4rem 0.8rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--surface-2);
    color: var(--text-muted) !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    text-decoration: none !important;
    transition: all 0.2s;
}
.dev-links a svg { width: 15px; height: 15px; flex-shrink: 0; }
.dev-links a:hover {
    color: var(--accent-glow) !important;
    border-color: var(--accent);
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(59,130,246,0.25);
}
.dev-links.compact a { padding: 0.45rem; border-radius: 8px; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface-2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Developer details ──────────────────────────────────────────────────────────
DEV_EMAIL    = "kumarajay3023ak3@gmail.com"
DEV_LINKEDIN = "https://www.linkedin.com/in/ajay-kumar-63a3b7326/"
DEV_GITHUB   = "https://github.com/Ajaykumar-02"

ICON_MAIL = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><polyline points="22,6 12,13 2,6"/></svg>'
ICON_LINKEDIN = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 1 1 0-4.125 2.062 2.062 0 0 1 0 4.125zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0z"/></svg>'
ICON_GITHUB = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>'

def dev_links_html(compact: bool = False) -> str:
    """Email / LinkedIn / GitHub icon links. compact=True shows icons only."""
    def link(href, icon, label, title):
        text = "" if compact else f"<span>{label}</span>"
        return f'<a href="{href}" target="_blank" rel="noopener noreferrer" title="{title}">{icon}{text}</a>'
    cls = "dev-links compact" if compact else "dev-links"
    return (
        f'<div class="{cls}">'
        + link(f"mailto:{DEV_EMAIL}", ICON_MAIL, "Email", "Email")
        + link(DEV_LINKEDIN, ICON_LINKEDIN, "LinkedIn", "LinkedIn")
        + link(DEV_GITHUB, ICON_GITHUB, "GitHub", "GitHub")
        + '</div>'
    )

# ─── API key check ──────────────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to your .env file (local) or to Secrets (Streamlit Cloud).")
    st.stop()


# ─── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-120b",
        temperature=0.4,
    )


def build_retriever(pdf_bytes: bytes):
    """Build a FAISS retriever from PDF bytes.
    Deliberately NOT cached with st.cache_resource: that cache is shared by every visitor,
    so a user's document could stay in server memory after they delete it."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name
    try:
        documents = PyPDFLoader(path).load()
    finally:
        os.remove(path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No readable text found in this PDF (it may be a scanned image).")

    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def sync_document(kind: str, uploaded):
    """Index the uploaded PDF once per session and keep it in THIS session only.
    No file uploaded -> the stored document is dropped. Returns (retriever, error)."""
    docs = st.session_state.docs
    if uploaded is None:
        docs[kind] = None
        return None, None

    pdf_bytes = uploaded.getvalue()
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    entry = docs[kind]
    if entry and entry["hash"] == digest:
        return entry["retriever"], None          # same file, already indexed

    try:
        with st.spinner(f"Indexing {uploaded.name}..."):
            retriever = build_retriever(pdf_bytes)
    except Exception as e:
        docs[kind] = None
        return None, str(e)

    docs[kind] = {"hash": digest, "name": uploaded.name, "retriever": retriever}
    return retriever, None


def status_row(label: str, retriever, error: str | None):
    if error:
        dot, text = "dot-error", f"{label}: could not read PDF"
    elif retriever is not None:
        dot, text = "dot-done", label
    else:
        dot, text = "dot-pending", label
    st.markdown(
        f'<div class="status-bar"><div class="status-dot {dot}"></div><span>{text}</span></div>',
        unsafe_allow_html=True,
    )
    if error:
        st.caption(error)


# ─── LangGraph state ────────────────────────────────────────────────────────────
class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


def route_query(state: State):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"


def build_graph(academic_retriever, fee_retriever, llm):
    """Graph is built per question (it is cheap), so it always uses the PDFs currently loaded."""

    def classifier_node(state: State) -> dict:
        """Look at the latest user message and decide which path to take."""
        last_message = state['messages'][-1].content

        prompt = (
            "Classify the following student query into exactly one category: "
            "'academic', 'fee', or 'general'.\n\n"
            "Use 'academic' for questions about attendance, exams, grading, credits, "
            "promotion, course structure, summer training, or degree requirements.\n"
            "Use 'fee' for questions about tuition, payment, refund, late charges, "
            "scholarships, or any money-related topic.\n"
            "Use 'general' for greetings, casual talk, or anything not related to "
            "the college rules or fee.\n\n"
            f"Query: {last_message}\n\n"
            "Return only one word: academic, fee, or general."
        )

        response = llm.invoke(prompt)
        category = response.content.strip().lower()

        if "academic" in category:
            category = "academic"
        elif "fee" in category:
            category = "fee"
        else:
            category = "general"

        return {"query_type": category}

    def academic_rag_node(state: State) -> dict:
        """Retrieves relevant chunks from the academics handbook."""
        if academic_retriever is None:
            return {"retrieved_context": "MISSING_DOC:Academic Handbook"}
        query = state["messages"][-1].content
        docs = academic_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def fee_rag_node(state: State) -> dict:
        """Retrieves relevant chunks from the fee structure PDF."""
        if fee_retriever is None:
            return {"retrieved_context": "MISSING_DOC:Fee Structure"}
        query = state["messages"][-1].content
        docs = fee_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def general_node(state: State) -> dict:
        """Answers directly using the LLM's own knowledge, no retrieval needed."""
        return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

    def response_node(state: State) -> dict:
        """Generates the final answer, personalized using the student's programme."""
        query = state["messages"][-1].content
        programme = state.get("programme", "Unknown")
        context = state["retrieved_context"]

        if context.startswith("MISSING_DOC:"):
            doc_name = context.split(":", 1)[1]
            message = (
                f"📎 The **{doc_name}** PDF has not been uploaded yet, so I can't answer "
                f"this from the official document. Please upload it from the sidebar "
                f"and ask again."
            )
            return {"messages": [("ai", message)]}

        if context == "NO_RETRIEVAL_NEEDED":
            prompt = (
                f"You are a friendly college assistant talking to a {programme} student. "
                f"Answer this question using your own general knowledge:\n\n{query}"
            )
        else:
            prompt = (
                f"You are a college assistant helping a {programme} student. "
                f"Use the following context from the official college documents to answer "
                f"the question accurately. If the context mentions specific figures for "
                f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Give a clear, friendly, and precise answer."
            )

        response = llm.invoke(prompt)
        return {"messages": [("ai", response.content.strip())]}

    graph = StateGraph(State)

    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_query)

    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")
    graph.add_edge("response", END)

    return graph.compile()


# ─── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []      # for display: {"role", "content", "query_type"}

if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []   # LangGraph message history


def clear_chat():
    st.session_state.messages = []
    st.session_state.lc_messages = []


if "docs" not in st.session_state:
    st.session_state.docs = {"academic": None, "fee": None}   # {"hash", "name", "retriever"}

if "upload_version" not in st.session_state:
    # changing the version changes the uploader's key, which empties the uploader
    st.session_state.upload_version = {"academic": 0, "fee": 0}


def remove_document(kind: str):
    """Delete one uploaded PDF and its search index from this session."""
    st.session_state.docs[kind] = None
    st.session_state.upload_version[kind] += 1
    gc.collect()


def delete_all_data():
    """Delete both PDFs, their indexes and the whole chat."""
    for kind in ("academic", "fee"):
        st.session_state.docs[kind] = None
        st.session_state.upload_version[kind] += 1
    st.session_state.messages = []
    st.session_state.lc_messages = []
    gc.collect()


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎓 College<br>Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Campus Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Setup</span>', unsafe_allow_html=True)
    programme_map = {
        "BCA": "BCA",
        "BBA": "BBA",
        "B.Com (H)": "B.Com (H)",
        "B.Tech CSE": "B.Tech CSE",
        "B.Tech AIML": "B.Tech AIML",
        "B.Tech EE": "B.Tech EE",
        "B.Tech ME": "B.Tech ME",
        "BA": "BA",
        "BSc": "BSc",
    }
    student_programme = st.selectbox(
        "Select your programme",
        options=list(programme_map.keys()),
        index=0,
    )

    st.markdown("---")

    st.markdown('<span class="badge badge-cyan">Knowledge Base</span>', unsafe_allow_html=True)

    academic_upload = st.file_uploader(
        "📘 Academic Handbook (RAG)",
        type=["pdf"],
        key=f"academic_upload_{st.session_state.upload_version['academic']}",
    )
    academic_retriever, academic_error = sync_document("academic", academic_upload)
    status_row(
        f"Uploaded: {academic_upload.name}" if academic_upload else "Not uploaded yet",
        academic_retriever, academic_error,
    )
    if academic_upload is not None:
        st.button("🗑️ Remove this file", key="remove_academic", use_container_width=True,
                  on_click=remove_document, args=("academic",), type="secondary")

    fee_upload = st.file_uploader(
        "💰 Fee Structure (RAG)",
        type=["pdf"],
        key=f"fee_upload_{st.session_state.upload_version['fee']}",
    )
    fee_retriever, fee_error = sync_document("fee", fee_upload)
    status_row(
        f"Uploaded: {fee_upload.name}" if fee_upload else "Not uploaded yet",
        fee_retriever, fee_error,
    )
    if fee_upload is not None:
        st.button("🗑️ Remove this file", key="remove_fee", use_container_width=True,
                  on_click=remove_document, args=("fee",), type="secondary")

    st.button("🗑️ Clear Chat", use_container_width=True, on_click=clear_chat, type="secondary")

    st.markdown("---")
    st.markdown('<span class="badge badge-purple">Privacy</span>', unsafe_allow_html=True)
    st.caption("Your PDFs are kept in memory for this session only and are never saved permanently.")
    st.button("🧹 Delete all my data", use_container_width=True, on_click=delete_all_data, type="secondary")
    st.caption("Removes your PDFs, their search index and the whole chat. Click it when you are done.")

    st.markdown("---")
    st.markdown('<span class="badge badge-green">Routing</span>', unsafe_allow_html=True)
    st.caption("📘 Academic Handbook (RAG)")
    st.caption("💰 Fee Structure (RAG)")
    st.caption("💬 General Knowledge")

    st.markdown("---")
    st.markdown('<div class="dev-credit">Developed by<br><span>Ajay Kumar</span></div>', unsafe_allow_html=True)
    st.markdown(dev_links_html(compact=True), unsafe_allow_html=True)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">College Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Academics · Fees · Campus help — for ' + student_programme + ' students</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Chat history ────────────────────────────────────────────────────────────────
if st.session_state.messages:
    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🎓"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant" and msg.get("query_type"):
                st.markdown(
                    f'<span class="badge badge-{msg["query_type"]}">{msg["query_type"]}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])
else:
    # Empty state
    missing = []
    if academic_retriever is None:
        missing.append("Academic Handbook")
    if fee_retriever is None:
        missing.append("Fee Structure")

    if len(missing) == 2:
        hint = "Upload the <strong>Academic Handbook</strong> and <strong>Fee Structure</strong> PDFs from the sidebar to start chatting."
    elif missing:
        hint = "Ask away! You can also upload the <strong>" + missing[0] + "</strong> PDF from the sidebar to get answers from it."
    else:
        hint = "Ask me about attendance, exams, fees, refunds, scholarships or anything campus-related."

    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎓</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Help
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:420px;line-height:1.7">
            {hint}
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-academic">Academic Handbook</span>
            <span class="badge badge-fee">Fee Structure</span>
            <span class="badge badge-general">General</span>
        </div>
    </div>""", unsafe_allow_html=True)

# ── Chat input ──────────────────────────────────────────────────────────────────
docs_ready = academic_retriever is not None or fee_retriever is not None
user_query = st.chat_input(
    "Type your question here..." if docs_ready else "Upload at least one PDF from the sidebar to start chatting",
    disabled=not docs_ready,
)

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_query)

    st.session_state.lc_messages.append(("human", user_query))

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking..."):
            try:
                graph_app = build_graph(academic_retriever, fee_retriever, get_llm())
                result = graph_app.invoke({
                    "programme": student_programme,
                    "messages": st.session_state.lc_messages,
                })
                ai_response = result["messages"][-1].content
                query_type = result.get("query_type", "general")

                st.markdown(
                    f'<span class="badge badge-{query_type}">{query_type}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(ai_response)

                st.session_state.lc_messages = result["messages"]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_response,
                    "query_type": query_type,
                })
            except Exception as e:
                # drop the unanswered question so the history stays consistent
                st.session_state.lc_messages.pop()
                st.session_state.messages.pop()
                st.error(f"Something went wrong: {e}")

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="dev-footer">Designed &amp; Developed by <span>Ajay Kumar</span>'
    + dev_links_html(compact=False)
    + '</div>',
    unsafe_allow_html=True,
)