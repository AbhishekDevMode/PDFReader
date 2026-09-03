import os
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader


load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
PREFERRED_CHAT_MODELS = (
    "models/gemini-3.5-flash",
    "models/gemini-3.5-flash-lite",
    "models/gemini-3-flash-preview",
    "models/gemini-flash-latest",
)
CHUNK_SIZE = 1_400
CHUNK_OVERLAP = 220


@dataclass
class DocumentChunk:
    """A searchable passage and the PDF page it came from."""

    text: str
    source: str
    page: int


def get_api_key() -> str | None:
    """Retrieve API key from Streamlit secrets, environment variables, or session state."""
    # 1. Check Streamlit Cloud secrets
    try:
        if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"]:
            return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass

    # 2. Check environment variable
    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # 3. Check user input in session state
    user_key = st.session_state.get("user_api_key")
    if user_key and str(user_key).strip():
        return str(user_key).strip()

    return None


def get_client() -> genai.Client:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Please enter your API key in the sidebar or add it to your .env file."
        )
    return genai.Client(api_key=api_key)


def get_chat_model(client: genai.Client) -> str:
    """Use an available generation model instead of a retired hard-coded name."""
    try:
        available = {
            model.name
            for model in client.models.list()
            if "generateContent" in (model.supported_actions or [])
        }
        for model_name in PREFERRED_CHAT_MODELS:
            if model_name in available:
                return model_name
        raise RuntimeError("No Gemini text-generation model is enabled for this API key.")
    except RuntimeError:
        raise
    except Exception:
        # The preferred current model provides a useful fallback if model listing is unavailable.
        return PREFERRED_CHAT_MODELS[0]


def split_text(text: str) -> list[str]:
    """Create overlapping, readable chunks without cutting words where possible."""
    clean_text = " ".join(text.split())
    if not clean_text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(clean_text):
        end = min(start + CHUNK_SIZE, len(clean_text))
        if end < len(clean_text):
            boundary = clean_text.rfind(" ", start + CHUNK_SIZE // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(clean_text[start:end])
        if end == len(clean_text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def read_pdfs(pdf_files: Iterable) -> tuple[list[DocumentChunk], list[str]]:
    chunks: list[DocumentChunk] = []
    warnings: list[str] = []
    for uploaded_file in pdf_files:
        try:
            reader = PdfReader(uploaded_file)
            extracted_any_text = False
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                for text in split_text(page_text):
                    chunks.append(DocumentChunk(text, uploaded_file.name, page_number))
                    extracted_any_text = True
            if not extracted_any_text:
                warnings.append(f"{uploaded_file.name}: no selectable text was found.")
        except Exception as error:
            warnings.append(f"{uploaded_file.name}: could not read this PDF ({error}).")
    return chunks, warnings


def embed_chunks(client: genai.Client, chunks: list[DocumentChunk]) -> np.ndarray:
    """Embed all passages in one API request and return unit-normalised vectors."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[chunk.text for chunk in chunks],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    vectors = np.asarray([embedding.values for embedding in response.embeddings], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def retrieve_context(client: genai.Client, question: str, limit: int = 6) -> list[tuple[DocumentChunk, float]]:
    if st.session_state.document_embeddings is None or len(st.session_state.document_chunks) == 0:
        return []

    question_response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query = np.asarray(question_response.embeddings[0].values, dtype=np.float32)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    scores = st.session_state.document_embeddings @ query

    actual_limit = min(limit, len(st.session_state.document_chunks))
    if actual_limit <= 0:
        return []

    best_indices = np.argsort(scores)[-actual_limit:][::-1]
    return [(st.session_state.document_chunks[index], float(scores[index])) for index in best_indices]


def answer_question(
    client: genai.Client, question: str, limit: int = 6
) -> tuple[str, list[tuple[DocumentChunk, float]]]:
    matches = retrieve_context(client, question, limit=limit)
    if not matches:
        return "I couldn't find any relevant passages in the processed documents.", []

    context = "\n\n".join(
        f"[Source: {chunk.source}, page {chunk.page}]\n{chunk.text}"
        for chunk, _ in matches
    )
    prompt = f"""You answer questions only from the supplied PDF excerpts.
If the excerpts do not contain the answer, say exactly: "I couldn't find that in the uploaded documents."
Do not rely on outside knowledge. Give a clear, helpful answer and cite source names and page numbers in square brackets.

PDF EXCERPTS:
{context}

QUESTION: {question}
"""
    response = client.models.generate_content(
        model=get_chat_model(client),
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1_000),
    )
    return response.text or "I couldn't generate an answer. Please try again.", matches


def generate_chat_export() -> str:
    """Format conversation history into Markdown for export."""
    lines = ["# PDF Compass — Chat Export", ""]
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"### {role}")
        lines.append(msg["content"])
        if msg.get("sources"):
            lines.append("")
            lines.append("**Sources used:**")
            for source in msg["sources"]:
                lines.append(f"- {source}")
        lines.append("")
    return "\n".join(lines)


def initialise_state() -> None:
    defaults = {
        "messages": [],
        "document_chunks": [],
        "document_embeddings": None,
        "document_names": [],
        "user_api_key": "",
        "top_k": 6,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(page_title="PDF Compass", page_icon="📚", layout="wide")
    initialise_state()

    st.markdown(
        """
    <style>
      .block-container { max-width: 1000px; padding-top: 2.25rem; }
      [data-testid="stSidebar"] { background: #101828; }
      [data-testid="stSidebar"] * { color: #f8fafc; }
      .hero { margin-bottom: 1.5rem; }
      .hero h1 { margin-bottom: .2rem; }
      .stDownloadButton button { width: 100%; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    api_key_configured = get_api_key() is not None

    with st.sidebar:
        st.title("📚 PDF Compass")
        st.caption("Grounded answers from your documents")

        # API Key management section
        if not api_key_configured:
            st.warning("🔑 Google API Key needed")
            user_key_input = st.text_input(
                "Enter Google API Key",
                type="password",
                placeholder="AIzaSy...",
                help="Get your key at https://aistudio.google.com/",
            )
            if user_key_input:
                st.session_state.user_api_key = user_key_input.strip()
                st.rerun()
        else:
            st.caption("🟢 Google API Key configured")

        uploaded_files = st.file_uploader(
            "Add PDF documents", type=["pdf"], accept_multiple_files=True
        )
        process = st.button(
            "Process documents",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files or not get_api_key(),
        )

        with st.expander("⚙️ RAG Settings"):
            st.session_state.top_k = st.slider(
                "Passages to retrieve per query",
                min_value=1,
                max_value=15,
                value=st.session_state.top_k,
                help="Number of document passages sent to Gemini for context.",
            )

        if st.session_state.document_chunks:
            st.divider()
            st.caption(
                f"Ready: {len(st.session_state.document_chunks)} passages from {len(st.session_state.document_names)} files"
            )
            for name in st.session_state.document_names:
                st.caption(f"• {name}")

            st.write("")
            if st.session_state.messages:
                st.download_button(
                    label="📥 Export Chat (.md)",
                    data=generate_chat_export(),
                    file_name="pdf_compass_chat.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            if st.button("Clear chat and documents", use_container_width=True):
                st.session_state.messages = []
                st.session_state.document_chunks = []
                st.session_state.document_embeddings = None
                st.session_state.document_names = []
                st.rerun()

    if process:
        try:
            with st.spinner("Reading and indexing your documents…"):
                chunks, warnings = read_pdfs(uploaded_files or [])
                if not chunks:
                    st.error("No readable text was found. Try a text-based PDF rather than a scanned image PDF.")
                else:
                    client = get_client()
                    st.session_state.document_embeddings = embed_chunks(client, chunks)
                    st.session_state.document_chunks = chunks
                    st.session_state.document_names = [file.name for file in uploaded_files or []]
                    st.session_state.messages = []
                    st.success(f"Ready to search {len(chunks)} passages across {len(uploaded_files or [])} document(s).")
                for warning in warnings:
                    st.warning(warning)
        except Exception as error:
            st.error(f"Could not process the documents: {error}")

    st.markdown(
        "<div class='hero'><h1>Ask your PDFs anything</h1><p>Upload documents, process them, then ask a question. Every answer is grounded in the uploaded text.</p></div>",
        unsafe_allow_html=True,
    )

    if not get_api_key():
        st.info("👈 Please enter your Google API Key in the sidebar to get started.")
        return

    if not st.session_state.document_chunks:
        st.info("Start by adding one or more PDF files in the sidebar and clicking 'Process documents'.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources used"):
                    for source in message["sources"]:
                        st.caption(source)

    if question := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents…"):
                try:
                    client = get_client()
                    response, matches = answer_question(
                        client, question, limit=st.session_state.top_k
                    )
                    st.markdown(response)
                    sources = list(
                        dict.fromkeys(f"{chunk.source} — page {chunk.page}" for chunk, _ in matches)
                    )
                    if sources:
                        with st.expander("Sources used"):
                            for source in sources:
                                st.caption(source)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response, "sources": sources}
                    )
                except Exception as error:
                    message = f"I couldn't answer that because of an API or document-search error: {error}"
                    st.error(message)
                    st.session_state.messages.append({"role": "assistant", "content": message})


if __name__ == "__main__":
    main()
