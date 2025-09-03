import chromadb
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_core.documents import Document
from typing import List


def get_vector_collection() -> chromadb.Collection:
    """
    Initializes or retrieves a ChromaDB collection configured with embedding functions.

    Returns:
        chromadb.Collection: A collection for storing and querying document embeddings.
    """
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: List[Document], file_name: str):
    """
    Adds document splits to the vector collection for semantic search.

    Args:
        all_splits: List of Document objects containing text chunks.
        file_name: Identifier for the uploaded file.
    """
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store!")


def query_collection(prompt: str, n_results: int = 10):
    """
    Queries the vector collection for documents relevant to the given prompt.

    Args:
        prompt: The query text.
        n_results: Number of results to retrieve. Defaults to 10.

    Returns:
        dict: Query results including documents and metadata.
    """
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results
