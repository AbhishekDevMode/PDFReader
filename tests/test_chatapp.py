import os
import numpy as np
import pytest
import streamlit as st
from chatapp import (
    DocumentChunk,
    get_api_key,
    split_text,
    generate_chat_export,
    initialise_state,
    retrieve_context,
)


def test_split_text_empty():
    assert split_text("") == []
    assert split_text("   ") == []


def test_split_text_short():
    sample_text = "This is a short paragraph of text."
    chunks = split_text(sample_text)
    assert len(chunks) == 1
    assert chunks[0] == sample_text


def test_split_text_long():
    words = ["word" + str(i) for i in range(500)]
    long_text = " ".join(words)
    chunks = split_text(long_text)
    assert len(chunks) > 1
    # Check that all original words are covered
    reconstructed = " ".join(chunks)
    assert "word0" in reconstructed
    assert "word499" in reconstructed


def test_document_chunk_dataclass():
    chunk = DocumentChunk(text="Sample content", source="test.pdf", page=3)
    assert chunk.text == "Sample content"
    assert chunk.source == "test.pdf"
    assert chunk.page == 3


def test_get_api_key_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test_env_key_123")
    assert get_api_key() == "test_env_key_123"


def test_get_api_key_session_state(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    st.session_state.user_api_key = "test_user_key_456"
    assert get_api_key() == "test_user_key_456"


def test_retrieve_context_empty():
    initialise_state()
    st.session_state.document_embeddings = None
    st.session_state.document_chunks = []
    # Mock client (not called when document_embeddings is None)
    matches = retrieve_context(None, "What is AI?", limit=5)
    assert matches == []


def test_generate_chat_export():
    initialise_state()
    st.session_state.messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {
            "role": "assistant",
            "content": "The capital of France is Paris.",
            "sources": ["geography.pdf — page 1"],
        },
    ]
    export_md = generate_chat_export()
    assert "# PDF Compass — Chat Export" in export_md
    assert "### User" in export_md
    assert "What is the capital of France?" in export_md
    assert "### Assistant" in export_md
    assert "The capital of France is Paris." in export_md
    assert "geography.pdf — page 1" in export_md
