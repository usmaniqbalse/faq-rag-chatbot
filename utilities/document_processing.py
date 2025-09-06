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
    # 1️⃣ Write uploaded file to a temp file and close it
    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name  # store path before closing

    try:
        # 2️⃣ Load PDF (PyMuPDFLoader will open/close internally)
        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()

        # 3️⃣ Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        return text_splitter.split_documents(docs)

    finally:
        # 4️⃣ Ensure temp file is deleted even if an error occurs
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except PermissionError:
                # If still locked, skip deletion to avoid crash
                pass
