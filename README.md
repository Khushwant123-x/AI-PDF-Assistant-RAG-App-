# 📄 RAG-Based PDF Question Answering System

A Retrieval-Augmented Generation (RAG) application that enables users to ask questions from PDF documents and receive context-aware answers grounded in the document content.

Built using LangChain, ChromaDB, Hugging Face Embeddings, and Mistral AI, the system retrieves the most relevant document chunks before generating responses, reducing hallucinations and improving answer accuracy.

---

## 🚀 Features

* PDF document ingestion and processing
* Semantic search using transformer-based embeddings
* Persistent vector storage with ChromaDB
* MMR (Maximal Marginal Relevance) retrieval for diversified context
* Context-grounded question answering
* Hallucination reduction through retrieval augmentation
* Interactive command-line chat interface

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Framework:** LangChain
* **LLM:** Mistral Small 2506
* **Embedding Model:** BAAI/bge-small-en-v1.5
* **Vector Database:** ChromaDB
* **Environment Management:** Python Dotenv

---

## 📂 System Architecture

1. Load PDF documents.
2. Split documents into manageable text chunks.
3. Generate vector embeddings using BGE embeddings.
4. Store embeddings in ChromaDB.
5. Retrieve relevant chunks using MMR search.
6. Inject retrieved context into a prompt template.
7. Generate grounded responses using Mistral AI.
8. Return answers to the user.

---

## 🔍 Retrieval Configuration

```python
search_type = "mmr"

search_kwargs = {
    "k": 4,
    "fetch_k": 10,
    "lambda_mult": 0.5
}
```

---

## 📦 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/rag-pdf-qa-system.git
cd rag-pdf-qa-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
MISTRAL_API_KEY=your_api_key
```

---

## ▶️ Run the Application

```bash
python rag.py
```

---

## 💻 Example Interaction

```text
You: What is Retrieval-Augmented Generation?

AI: Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with large language models to generate accurate, context-aware responses using external knowledge sources.
```

---

## 🎯 Future Enhancements

* Streamlit-based web interface
* FastAPI deployment
* Multi-PDF support
* Source citations and references
* Conversational memory
* Hybrid retrieval (BM25 + Semantic Search)
* Cloud deployment (AWS/GCP/Azure)

---

## 📈 Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Semantic Search
* Prompt Engineering
* LangChain Framework
* LLM Integration
* Embedding Models
* Information Retrieval

---

## 👨‍💻 Author

**Khushwant Singh Rajat**

Aspiring AI Engineer passionate about Generative AI, Retrieval-Augmented Generation (RAG), Machine Learning, and Large Language Model applications.

