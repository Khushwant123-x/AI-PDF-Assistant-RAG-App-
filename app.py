import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------------------------------------------------
# Page Configuration & Metadata
# ---------------------------------------------------------
st.set_page_config(
    page_title="DocuMind AI • Neural RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS (Glassmorphism & Obsidian Dark Theme)
# ---------------------------------------------------------
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* Global Font & Theme */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #E2E8F0;
}

/* Background gradient */
.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 45%),
                radial-gradient(circle at 85% 85%, rgba(14, 165, 233, 0.08) 0%, transparent 45%),
                linear-gradient(180deg, #07090e 0%, #0d1322 100%);
    background-attachment: fixed;
}

/* Hide Streamlit Header & Footer deco */
header[data-testid="stHeader"] {
    background: rgba(7, 9, 14, 0.6);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: rgba(13, 19, 34, 0.75);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Glass Card */
.glass-card {
    background: rgba(17, 24, 39, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 12px 35px -8px rgba(99, 102, 241, 0.15);
}

/* Hero Title */
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.85rem;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #818CF8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.3rem;
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #FFFFFF 30%, #A5B4FC 70%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.35rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #94A3B8;
    margin-bottom: 1.25rem;
}

/* Metrics Badges */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    margin-top: 0.75rem;
}
.stat-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 0.6rem 0.75rem;
    text-align: center;
}
.stat-num {
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
}
.stat-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-top: 0.15rem;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background: rgba(15, 23, 42, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.85rem !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 20px -5px rgba(0, 0, 0, 0.3) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(30, 41, 59, 0.5) !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
}

/* Chat Input Bar */
[data-testid="stChatInput"] {
    border-radius: 16px !important;
    background: rgba(17, 24, 39, 0.85) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #6366F1 !important;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.35) !important;
}

/* Button Customization */
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
}

/* Clear Button */
.clear-btn button {
    background: rgba(239, 68, 68, 0.1) !important;
    color: #F87171 !important;
    border: 1px solid rgba(239, 68, 68, 0.25) !important;
    box-shadow: none !important;
}
.clear-btn button:hover {
    background: rgba(239, 68, 68, 0.2) !important;
    color: #FECACA !important;
}

/* Quick prompt chips */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 1rem 0 1.5rem 0;
}
.prompt-chip {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #CBD5E1;
    padding: 0.45rem 0.9rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.2s ease;
}
.prompt-chip:hover {
    background: rgba(99, 102, 241, 0.2);
    border-color: #818CF8;
    color: #FFFFFF;
}

/* Citations Accordion */
[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    margin-top: 0.6rem !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Caching: Embedding Model
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Neural Embeddings...")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

embedding_model = load_embedding_model()

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {"pages": 0, "chunks": 0, "name": "None"}
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# ---------------------------------------------------------
# Sidebar: Brand, Settings & PDF Ingestion
# ---------------------------------------------------------
with st.sidebar:
    # App Logo & Branding
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.25rem;">
            <div style="font-size: 2rem; background: linear-gradient(135deg, #6366F1, #38BDF8); width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 12px; box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35);">
                🧠
            </div>
            <div>
                <h2 style="font-family: 'Outfit'; margin: 0; font-size: 1.25rem; font-weight: 700; color: #FFFFFF;">DocuMind AI</h2>
                <span style="font-size: 0.75rem; color: #94A3B8; letter-spacing: 0.04em;">INTELLIGENT RAG ENGINE</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Document Metrics Card
    stats = st.session_state.doc_stats
    st.markdown(f"""
        <div class="glass-card">
            <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; margin-bottom: 0.5rem;">
                📊 Active Document Status
            </div>
            <div style="font-weight: 600; color: #F1F5F9; font-size: 0.88rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                📄 {stats['name']}
            </div>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-num">{stats['pages']}</div>
                    <div class="stat-label">Pages</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num">{stats['chunks']}</div>
                    <div class="stat-label">Chunks</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num">{'Ready' if stats['chunks'] > 0 else 'Idle'}</div>
                    <div class="stat-label">Index</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Upload Section
    st.markdown("### 📁 Upload Document")
    pdf_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload text-based PDF documents for semantic indexing",
        label_visibility="collapsed"
    )

    if pdf_file and st.session_state.last_uploaded_file != pdf_file.name:
        with st.spinner("⚡ Extracting & Indexing document chunks..."):
            file_path = "temp.pdf"
            with open(file_path, "wb") as f:
                f.write(pdf_file.read())

            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(docs)
                valid_chunks = [c for c in chunks if c.page_content.strip()]

                if not valid_chunks:
                    st.error("⚠️ Is PDF se koi readable text extract nahi ho paya. (Ye scanned/image-only PDF lagti hai). Kripya selectable text wali PDF use karein.")
                    st.session_state.vectorstore = None
                    st.session_state.doc_stats = {"pages": len(docs), "chunks": 0, "name": pdf_file.name}
                else:
                    vectorstore = Chroma.from_documents(
                        documents=valid_chunks,
                        embedding=embedding_model,
                        persist_directory="chroma_db"
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.last_uploaded_file = pdf_file.name
                    st.session_state.doc_stats = {
                        "pages": len(docs),
                        "chunks": len(valid_chunks),
                        "name": pdf_file.name
                    }
                    st.success(f"✅ Indexed {len(valid_chunks)} chunks from {len(docs)} pages!")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    st.markdown("---")

    # Model & Engine Controls
    st.markdown("### ⚙️ Engine Settings")
    selected_model = st.selectbox(
        "Groq LLM Model",
        options=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
        index=0,
        help="High-speed Groq-hosted open foundation models"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Lower values yield more factual, strict answers based solely on context."
    )

    st.markdown("---")

    # API Key retrieval with fallback to st.secrets or user input
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]

    if not groq_api_key:
        groq_api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Enter your Groq API key (or set GROQ_API_KEY in .env / secrets)"
        )

    # Clear chat session
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# LLM & Prompt Setup
# ---------------------------------------------------------
llm = ChatGroq(
    model=selected_model,
    temperature=temperature,
    api_key=groq_api_key if groq_api_key else "dummy_key"
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are DocuMind AI, an intelligent, precise, and articulate research assistant.\n"
     "RULES:\n"
     "1. Answer the question thoroughly based ONLY on the provided Context.\n"
     "2. Use structured Markdown, bullet points, and bold text for clarity.\n"
     "3. If the answer cannot be found in the context, politely state: "
     "'I could not find relevant details regarding this in the uploaded document.' Do not invent information.\n"
     "4. Maintain a professional, executive tone."
    ),
    ("human",
     "Context:\n{context}\n\nQuestion:\n{question}")
])


# ---------------------------------------------------------
# Main Chat Area: Hero Header
# ---------------------------------------------------------
st.markdown("""
    <div class="hero-badge">⚡ Ultra-Fast Groq Inference • BGE Embeddings</div>
    <div class="hero-title">Ask Anything About Your Documents</div>
    <div class="hero-sub">Upload contracts, research papers, lecture slides, or manuals and converse with them in real-time.</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Empty State: Quick Prompt Suggestions
# ---------------------------------------------------------
if not st.session_state.messages:
    st.markdown("""
        <div class="glass-card" style="margin-top: 1rem; border-left: 4px solid #6366F1;">
            <div style="font-weight: 600; font-size: 1rem; color: #FFFFFF; margin-bottom: 0.35rem;">
                🚀 Getting Started
            </div>
            <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.5;">
                1. Upload a PDF using the left sidebar.<br>
                2. Once indexed, type your question below or click one of the suggested prompts to begin.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin-bottom: 0.5rem;">
            💡 Quick Prompt Ideas
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Summarize Document", use_container_width=True):
            st.session_state.pending_query = "Please provide an executive summary of the entire document with key takeaways."
    with col2:
        if st.button("🔍 Key Insights & Facts", use_container_width=True):
            st.session_state.pending_query = "What are the most critical insights, facts, and figures mentioned in this document?"
    with col3:
        if st.button("❓ Top Questions & Answers", use_container_width=True):
            st.session_state.pending_query = "Formulate the top 5 questions that this document answers, along with brief answers."


# ---------------------------------------------------------
# Render Message History
# ---------------------------------------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "👤" if role == "user" else "🧠"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"🔎 Source References ({len(msg['sources'])} chunks retrieved)"):
                for idx, src in enumerate(msg["sources"], 1):
                    page_num = src.get("page", "Unknown")
                    st.markdown(f"**Chunk {idx} (Page {page_num}):**")
                    st.caption(src.get("text", ""))


# ---------------------------------------------------------
# Chat Input & Retrieval Pipeline
# ---------------------------------------------------------
query = st.chat_input("Ask a question about the document...")

# Handle button clicks from quick prompt suggestions
if "pending_query" in st.session_state and st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

if query:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Kripya pehle sidebar se ek readable PDF upload karein!")
    else:
        # 1. Append & Display User Message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        # 2. Assistant Thinking & Retrieval
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Searching document & formulating answer..."):
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 4,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )

                docs = retriever.invoke(query)
                context = "\n\n".join([d.page_content for d in docs])

                final_prompt = prompt.invoke({
                    "context": context,
                    "question": query
                })

                response = llm.invoke(final_prompt)
                answer = response.content

                # Format source citations
                sources = [
                    {
                        "page": d.metadata.get("page", 0) + 1 if "page" in d.metadata else "N/A",
                        "text": d.page_content[:280] + ("..." if len(d.page_content) > 280 else "")
                    }
                    for d in docs
                ]

                st.markdown(answer)
                if sources:
                    with st.expander(f"🔎 Source References ({len(sources)} chunks retrieved)"):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(f"**Chunk {idx} (Page {src['page']}):**")
                            st.caption(src["text"])

                # 3. Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })