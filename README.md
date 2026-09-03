# 📚 PDF Compass — Multi-PDF Chat Application with Google Gemini

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.44%2B-red.svg)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-v1.0%2B-4285F4.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**PDF Compass** is an interactive, production-ready Retrieval-Augmented Generation (RAG) web application built with Python, Streamlit, and Google Gemini APIs. It allows users to upload multiple PDF documents, automatically index and embed document passages, and ask natural language questions with strict grounding and source citations (file name + page numbers).

---

## 🏗️ System Architecture

![Architecture Flow](img/Architecture.jpg)

### How It Works:
1. **Document Ingestion & Processing**: Uploaded PDF files are parsed page-by-page using `pypdf`.
2. **Text Chunking**: Text is split into overlapping passages (`1400` characters with `220` overlap) to retain context across boundaries.
3. **Vector Embeddings**: Passages are embedded using Google's `gemini-embedding-001` model in unit-normalized vector representations.
4. **Context Retrieval**: User queries are embedded in real time (`RETRIEVAL_QUERY`), and cosine similarity matches top-K document passages.
5. **Grounded Generation**: Gemini generates answers strictly based on retrieved passages, citing exact source files and page numbers.

---

## ✨ Features

- **Multi-PDF Processing**: Upload and query multiple documents simultaneously.
- **Strictly Grounded Answers**: Prevents hallucinations by forcing answers to come only from uploaded documents.
- **Source Citations**: Every answer includes expandable citations showing exact source PDF names and page numbers.
- **Flexible API Key Setup**: Load keys automatically from environment variables (`.env`), Streamlit Secrets, or directly via the UI sidebar.
- **RAG Fine-Tuning**: Adjust the top-K retrieved passage count dynamically using a sidebar slider.
- **Chat History Export**: Export complete conversations with source citations as a downloadable Markdown (`.md`) file.
- **Deployment Ready**: Fully configured with Docker, `.dockerignore`, Streamlit config, and automated unit test suite.

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- Python 3.10 or higher
- A Google Gemini API Key ([Get your key from Google AI Studio](https://aistudio.google.com/))

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/Multi-PDFs_ChatApp_AI-Agent.git
cd Multi-PDFs_ChatApp_AI-Agent

# Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and add your API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GOOGLE_API_KEY=your_google_api_key_here
```
*(Note: If no API key is present in `.env`, you can enter it directly in the application sidebar when running.)*

### 5. Launch the Application
```bash
streamlit run chatapp.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the test suite to verify text chunking, state management, and retrieval logic:
```bash
pytest tests/
```

---

## 🐳 Deployment Guide

### Option A: Streamlit Community Cloud (Recommended)

1. Push your repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New App**.
3. Select your repository, branch (`main`), and main file path (`chatapp.py`).
4. In **Advanced Settings -> Secrets**, add your API key:
   ```toml
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```
5. Click **Deploy**!

---

### Option B: Docker Container (Render, Cloud Run, Railway, HuggingFace)

A production-ready `Dockerfile` is included in the repository.

#### Build Docker Image:
```bash
docker build -t pdf-compass .
```

#### Run Docker Container:
```bash
docker run -p 8501:8501 -e GOOGLE_API_KEY="your_google_api_key_here" pdf-compass
```
Access the application at `http://localhost:8501`.

---

## 📁 Directory Structure

```
Multi-PDFs_ChatApp_AI-Agent/
├── chatapp.py              # Main Streamlit application logic
├── requirements.txt        # Python dependencies
├── Dockerfile              # Containerization instructions
├── .dockerignore           # Excluded files for Docker build
├── .env.example            # Environment template file
├── .gitignore              # Git ignore rules (protects API keys)
├── .streamlit/
│   └── config.toml         # Streamlit server and theme settings
├── tests/
│   └── test_chatapp.py     # Unit test suite
├── docs/                   # Sample PDF documents for testing
├── img/                    # Architecture diagrams & media
└── README.md               # Documentation
```

---

## 🛡️ Security Best Practices

- Never commit `.env` files or API keys to version control. `.env` is listed in `.gitignore`.
- API keys entered via the UI sidebar are saved strictly in `st.session_state` (in-memory per session) and are never logged or saved to disk.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
