# pipeline.py

from rag.src.loader import DocumentLoader
from rag.src.cuad_loader import CUADLoader
from rag.src.chunking import Chunking
from rag.src.vector import Vectorizer
from rag.src.retrieval import Retriever
from rag.src.memory import Memory
from rag.src.generation import Generator
from rag.src.router import IntentRouter                        # NEW
from agents.legal_rag_agent.qa_agent import QAAgent            # NEW

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "gemini-3.1-flash-lite"
API_KEY = os.getenv("GEMINI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
USER_DATA_DIR = BASE_DIR / "user_data"
CHROMA_DIR = BASE_DIR / "chroma_db"
CUAD_SAMPLE = DATA_DIR / "cuad_sample.json"


class RAGPipeline:

    def __init__(self):
        self.model_name = GENERATION_MODEL
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.chroma_dir = CHROMA_DIR
        self.api_key = API_KEY

        self.memory = Memory()
        self.generator = Generator(model_name=GENERATION_MODEL, api_key=API_KEY)

        # NEW: the intent router (decides document vs CUAD for the 3 use cases)
        self.router = IntentRouter(self.generator)

        self.knowledge_retriever = None    # CUAD
        self.document_retriever = None     # user upload

    # ---------------- CUAD knowledge base ----------------
    def ingest_cuad(self):
        loader = CUADLoader(sample_path=CUAD_SAMPLE)
        documents = loader.load_sample()
        chunker = Chunking(documents, self.embedding_model)
        chunks = chunker.semantic_chunking()
        vectorizer = Vectorizer(self.chroma_dir, self.embedding_model,
                                collection_name="legal_knowledge")
        vector_store = vectorizer.build_vector(chunks)
        retriever_obj = vectorizer.create_retriever(vector_store)
        self.knowledge_retriever = Retriever(retriever_obj)
        print("CUAD knowledge base ingested")

    # ---------------- User uploaded document (from OCR text) ----------------
    def ingest_document_text(self, text: str, source_name: str = "uploaded_document"):
        if not text or not text.strip():
            print("No text to ingest.")
            return 0
        documents = [Document(
            page_content=text,
            metadata={"source_file": source_name, "source_name": source_name, "source_index": 1},
        )]
        chunker = Chunking(documents, self.embedding_model)
        chunks = chunker.semantic_chunking()
        vectorizer = Vectorizer(self.chroma_dir, self.embedding_model,
                                collection_name="user_documents")
        vector_store = vectorizer.build_vector(chunks)
        retriever_obj = vectorizer.create_retriever(vector_store)
        self.document_retriever = Retriever(retriever_obj)
        print(f"User document ingested (OCR text): {len(chunks)} chunks")
        return len(chunks)

    # ---------------- get retrievers ----------------
    def _get_knowledge_retriever(self):
        if self.knowledge_retriever is None:
            vectorizer = Vectorizer(self.chroma_dir, self.embedding_model,
                                    collection_name="legal_knowledge")
            if not Path(CHROMA_DIR).exists():
                self.ingest_cuad()
            else:
                vector_store = vectorizer.load_vector_store()
                retriever_obj = vectorizer.create_retriever(vector_store)
                self.knowledge_retriever = Retriever(retriever_obj)
        return self.knowledge_retriever

    def _get_document_retriever(self):
        if self.document_retriever is None:
            vectorizer = Vectorizer(self.chroma_dir, self.embedding_model,
                                    collection_name="user_documents")
            vector_store = vectorizer.load_vector_store()
            retriever_obj = vectorizer.create_retriever(vector_store)
            self.document_retriever = Retriever(retriever_obj)
        return self.document_retriever

    # ---------------- get the final answer (uses QA Agent) ----------------
    def get_answer(self, question: str, has_document: bool = False, general: bool = False) -> dict:
        # make sure retrievers exist
        self._get_knowledge_retriever()
        if has_document:
            self._get_document_retriever()

        # build the QA Agent with the current retrievers + router
        qa_agent = QAAgent(
            knowledge_retriever=self.knowledge_retriever,
            document_retriever=self.document_retriever,
            generator=self.generator,
            memory=self.memory,
            router=self.router,
        )

        # the QA Agent answers (it uses the router internally)
        return qa_agent.answer(question, has_document, general)

    def clear_memory(self):
        self.memory.clear()