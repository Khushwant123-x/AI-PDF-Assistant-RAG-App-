# 📄 RAG-Based PDF Question Answering System

A Retrieval-Augmented Generation (RAG) application built using LangChain, ChromaDB, Hugging Face Embeddings, and Mistral AI. The system retrieves relevant information from indexed PDF documents and generates accurate answers based solely on the document content.

## 🚀 Features

* PDF document ingestion and indexing
* Semantic search using vector embeddings
* ChromaDB vector storage
* MMR (Maximal Marginal Relevance) retrieval
* Context-aware question answering
* Hallucination reduction through document grounding
* Command-line chat interface
* Fast document retrieval and response generation

---

## 🛠️ Tech Stack

### Frameworks

* LangChain
* LangChain Community

### LLM

* Mistral Small 2506

### Embedding Model

* BAAI/bge-small-en-v1.5

### Vector Database

* ChromaDB

### Environment Management

* Python Dotenv

---

## ⚙️ Architecture

```text
User Question
      │
      ▼
Chroma Retriever (MMR Search)
      │
      ▼
Relevant Document Chunks
      │
      ▼
Prompt Template
      │
      ▼
Mistral AI
      │
      ▼
Generated Answer
```

---

## 📂 Project Workflow

1. Load environment variables.
2. Connect to the persisted ChromaDB vector store.
3. Generate embeddings using BAAI/bge-small-en-v1.5.
4. Retrieve relevant document chunks using MMR retrieval.
5. Inject retrieved context into a prompt template.
6. Send context and query to Mistral AI.
7. Return a grounded answer based only on the document content.

---

## 🔍 Retrieval Configuration

```python
search_type = "mmr"

k = 4
fetch_k = 10
lambda_mult = 0.5
```

This configuration improves diversity in retrieved document chunks while maintaining relevance.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/rag-pdf-qa-system.git
cd rag-pdf-qa-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
MISTRAL_API_KEY=your_api_key
```

Run the application:

```bash
python rag.py
```

---

## 💬 Example

```text
You: What is Retrieval-Augmented Generation?

AI: Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with large language models to generate more accurate and context-aware responses.
```

---

## 🎯 Key Concepts Implemented

* Retrieval-Augmented Generation (RAG)
* Vector Embeddings
* Semantic Search
* Chroma Vector Database
* MMR Retrieval
* Prompt Engineering
* Context Grounding
* Large Language Models

---

## 📈 Future Improvements

* Streamlit Web Interface
* Multi-PDF Support
* Source Citations
* Conversational Memory
* Hybrid Search
* FastAPI Deployment
* Cloud Deployment

---

## 👨‍💻 Author

Khushwant Singh Rajat

Built to explore Retrieval-Augmented Generation (RAG), Vector Databases, and Large Language Models using the LangChain ecosystem.
