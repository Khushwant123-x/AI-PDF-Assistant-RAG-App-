# 🧠 DocuMind AI • Neural RAG Assistant

A state-of-the-art Retrieval-Augmented Generation (RAG) assistant built with **Streamlit**, **LangChain**, **Groq LLM inference**, and **Chroma Vector Store**.

![UI Preview](https://img.shields.io/badge/UI-Obsidian%20Glass-6366f1)
![Embeddings](https://img.shields.io/badge/Embeddings-BAAI%2Fbge--small--en--v1.5-38bdf8)
![LLM](https://img.shields.io/badge/LLM-Groq%20Ultra--Fast-10b981)

---

## ⚡ Features

- **Blazing Fast Responses:** Powered by Groq's high-throughput LPU inference (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`).
- **Dense Semantic Retrieval:** Powered by `BAAI/bge-small-en-v1.5` normalized embeddings with MMR (Maximal Marginal Relevance) re-ranking.
- **Glassmorphic Obsidian Dark UI:** Custom responsive styling with Google Fonts (*Outfit* & *Plus Jakarta Sans*), animated metrics, and quick prompt suggestions.
- **Source Citations:** Every response includes collapsible chunk references and page numbers for auditability.
- **Production Ready:** Pre-configured for Streamlit Community Cloud and Docker deployment.

---

## 🚀 Quick Start (Local)

1. **Clone repository:**
   ```bash
   git clone <your-repo-url>
   cd "RAG Project"
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables:**
   Create a `.env` file in the root folder:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment Guide (Streamlit Community Cloud - Free)

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Deploy DocuMind AI RAG App"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and click **"New app"**.
   - Select your GitHub repository, branch (`main`), and set Main file path to: `app.py`.
   - Click **"Advanced settings..."** -> **Secrets** and add:
     ```toml
     GROQ_API_KEY = "your_actual_groq_api_key"
     ```
   - Click **Deploy!** 🚀
