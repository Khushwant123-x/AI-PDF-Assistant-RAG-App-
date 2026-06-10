import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="RAG PDF Assistant", page_icon="📚", layout="wide")

st.title("📚 AI PDF Assistant (RAG App)")
st.write("Upload PDF and chat with it 🤖")

# -----------------------
# Embeddings
# -----------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# -----------------------
# LLM
# -----------------------
llm = ChatMistralAI(model="mistral-small-2506")

# -----------------------
# Prompt
# -----------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant. "
     "Use ONLY the provided context. "
     "If answer is not present, say: 'I could not find the answer in the document.'"
    ),
    ("human",
     "Context:\n{context}\n\nQuestion:\n{question}")
])

# -----------------------
# Session State
# -----------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------
# Sidebar - Upload PDF
# -----------------------
st.sidebar.header("📁 Upload PDF")

pdf_file = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])

if pdf_file:
    with st.spinner("Processing PDF..."):

        file_path = "temp.pdf"
        with open(file_path, "wb") as f:
            f.write(pdf_file.read())

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        st.session_state.vectorstore = vectorstore

        st.sidebar.success("PDF processed successfully ✅")

# -----------------------
# Chat UI (Clean ChatGPT Style)
# -----------------------
st.subheader("💬 Chat with PDF")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
query = st.chat_input("Ask something from your PDF...")

if query:

    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF first.")
    else:

        # Save user message
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

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

        # Save AI response
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)