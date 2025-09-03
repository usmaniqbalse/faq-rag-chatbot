import os
import tempfile

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Local modules (pure organization; no behavior changes)
from utilities.document_processing import process_document
from utilities.vector_store import (
    get_vector_collection,
    add_to_vector_collection,
    query_collection,
)
from utilities.model_inference import call_llm


def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    """
    Re-ranks documents using a cross-encoder model for relevance.

    Args:
        documents: List of document strings to rank.
        prompt: Query prompt to determine relevance.

    Returns:
        tuple: Concatenated text of top-ranked documents and their indices.
    """
   
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids


if __name__ == "__main__":
    # ─────────────────────────────────────────────────────────────────────────
    # Sidebar: upload + ingestion 
    # ─────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        # Kept inside sidebar to match your original behavior
        st.set_page_config(page_title="RAG Question Answer")
        uploaded_file = st.file_uploader(
            "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        )

        process = st.button("⚡️ Process")
        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    # ─────────────────────────────────────────────────────────────────────────
    # Main content: ask + answer 
    # ─────────────────────────────────────────────────────────────────────────
    st.header("🗣️ RAG Question Answer")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button("🔥 Ask")

    if ask and prompt:
        results = query_collection(prompt)
        context = results.get("documents")[0]
        relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
        response = call_llm(context=relevant_text, prompt=prompt)
        st.write_stream(response)

        with st.expander("See retrieved documents"):
            st.write(results)

        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)
