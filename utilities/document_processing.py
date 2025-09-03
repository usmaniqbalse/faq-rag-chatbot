# File ingestion + chunking. 
import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile


def process_document(uploaded_file: UploadedFile) -> List[Document]:
    """
    Processes an uploaded PDF file by splitting it into text chunks for analysis.

    Args:
        uploaded_file: A Streamlit UploadedFile object containing the PDF file.

    Returns:
        A list of Document objects containing chunked text from the PDF.
    """
    # Store uploaded file as a temp file (deleted immediately after load)
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)  # Delete temp file to avoid leakage

    # Conservative chunks to maximize retrieval signal
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)
