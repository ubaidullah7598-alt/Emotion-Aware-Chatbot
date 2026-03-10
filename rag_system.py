"""
rag_system.py
─────────────────────────────────────────────────
RAG pipeline: ingest PDFs / text files → chunk →
embed → FAISS vector store → similarity retrieval.
"""

from __future__ import annotations
import os, tempfile
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ── Embeddings (free, local) ───────────────────
def _get_embeddings():
    """Return HuggingFace embeddings (no API key needed)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# ── Document loader ────────────────────────────
def load_document(file_path: str) -> List[Document]:
    """Load a PDF or plain-text file and return LangChain Documents."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


# ── Chunking ───────────────────────────────────
def chunk_documents(docs: List[Document], chunk_size: int = 800,
                    chunk_overlap: int = 100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


# ── Vector store ───────────────────────────────
class RAGSystem:
    """Wrapper around FAISS for multi-document, in-memory RAG."""

    def __init__(self):
        self.vectorstore: Optional[FAISS] = None
        self.embeddings = _get_embeddings()
        self.loaded_files: List[str] = []

    # ── ingest ──────────────────────────────────
    def add_document(self, uploaded_file) -> str:
        """
        Accept a Streamlit UploadedFile, persist to a temp path,
        chunk, embed, and add to the FAISS index.
        Returns a status message.
        """
        suffix = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            raw_docs = load_document(tmp_path)
            chunks   = chunk_documents(raw_docs)
            if not chunks:
                return f"⚠️  No text found in **{uploaded_file.name}**."

            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vectorstore.add_documents(chunks)

            self.loaded_files.append(uploaded_file.name)
            return (f"✅ **{uploaded_file.name}** ingested — "
                    f"{len(chunks)} chunks indexed.")
        except Exception as exc:
            return f"❌ Error processing **{uploaded_file.name}**: {exc}"
        finally:
            os.unlink(tmp_path)

    # ── retrieve ────────────────────────────────
    def retrieve(self, query: str, k: int = 4) -> str:
        """Return top-k relevant passages as a single string."""
        if self.vectorstore is None:
            return ""
        results: List[Document] = self.vectorstore.similarity_search(query, k=k)
        if not results:
            return ""
        passages = [f"[Source: {d.metadata.get('source', 'doc')} "
                    f"p.{d.metadata.get('page', '?')}]\n{d.page_content}"
                    for d in results]
        return "\n\n---\n\n".join(passages)

    # ── helpers ─────────────────────────────────
    @property
    def has_documents(self) -> bool:
        return self.vectorstore is not None

    def clear(self):
        self.vectorstore  = None
        self.loaded_files = []
