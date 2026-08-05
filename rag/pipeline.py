# pipeline.py

from rag.src.loader import DocumentLoader
from rag.src.cuad_loader import CUADLoader
from rag.src.chunking import Chunking
from rag.src.vector import Vectorizer
from rag.src.retrieval import Retriever
from rag.src.memory import Memory
from rag.src.generation import Generator

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "gemini-2.0-flash"
API_KEY = os.getenv("GEMINI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"                 # CUAD knowledge base source
USER_DATA_DIR = BASE_DIR / "user_data"       # the user's uploaded document(s)
CHROMA_DIR = BASE_DIR / "chroma_db"
CUAD_SAMPLE = DATA_DIR / "cuad_sample.json"


class RAGPipeline:
    """
    Legal Assistant RAG pipeline with THREE use cases:
      1. No document uploaded            -> answer from CUAD  (legal_knowledge)
      2. Document uploaded, ask about it -> answer from the uploaded doc (user_documents)
      3. Document uploaded, general Q     -> answer from CUAD  (legal_knowledge)

    'has_document' tells the pipeline whether a document is currently uploaded.
    'general' tells the pipeline the user is asking a general legal question
    (so even with a document uploaded, we use CUAD).

    Document text can be ingested TWO ways:
      - ingest_document()       -> loads raw files from user_data/ (digital only)
      - ingest_document_text()  -> uses text ALREADY extracted by the Day-1 OCR
                                   agent (RECOMMENDED - handles scanned docs too)
    """

    def __init__(self):

        self.model_name = GENERATION_MODEL
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        self.chroma_dir = CHROMA_DIR
        self.api_key = API_KEY

        self.memory = Memory()

        self.generator = Generator(
            model_name=GENERATION_MODEL,
            api_key=API_KEY,
        )

        # two separate retrievers (built lazily)
        self.knowledge_retriever = None    # CUAD
        self.document_retriever = None     # user upload

    # ---------------- CUAD knowledge base (one-time) ----------------
    def ingest_cuad(self):

        # load CUAD from the local sample json (use load_from_huggingface for full)
        loader = CUADLoader(sample_path=CUAD_SAMPLE)
        documents = loader.load_sample()

        chunker = Chunking(documents, self.embedding_model)
        chunks = chunker.semantic_chunking()

        vectorizer = Vectorizer(
            self.chroma_dir,
            self.embedding_model,
            collection_name="legal_knowledge",
        )
        vector_store = vectorizer.build_vector(chunks)

        retriever_obj = vectorizer.create_retriever(vector_store)
        self.knowledge_retriever = Retriever(retriever_obj)
        print("CUAD knowledge base ingested")

    # ---------------- User uploaded document (raw file loader) ----------------
    def ingest_document(self):
        """
        Loads raw files (PDF/DOCX/TXT) from user_data/ using LangChain loaders.
        NOTE: LangChain's PyPDFLoader does NOT do OCR, so scanned PDFs/images
        will not work here. For scanned documents, use ingest_document_text()
        with the text from the Day-1 OCR agent instead.
        """
        loader = DocumentLoader(USER_DATA_DIR)
        documents = loader.load_documents()

        chunker = Chunking(documents, self.embedding_model)
        chunks = chunker.semantic_chunking()

        vectorizer = Vectorizer(
            self.chroma_dir,
            self.embedding_model,
            collection_name="user_documents",
        )
        vector_store = vectorizer.build_vector(chunks)

        retriever_obj = vectorizer.create_retriever(vector_store)
        self.document_retriever = Retriever(retriever_obj)
        print("User document ingested (raw file)")

    # ---------------- User uploaded document (from Day-1 OCR text) ----------------
    def ingest_document_text(self, text: str, source_name: str = "uploaded_document"):
        """
        RECOMMENDED for user uploads.
        Ingests the ALREADY-EXTRACTED, cleaned text from the Day-1 OCR agent
        (extract_text). This reuses Vasanth's OCR output, so it handles scanned
        PDFs and images too, and avoids extracting the document twice.
        """
        if not text or not text.strip():
            print("No text to ingest.")
            return 0

        # wrap the extracted text as a LangChain Document
        documents = [
            Document(
                page_content=text,
                metadata={
                    "source_file": source_name,
                    "source_name": source_name,
                    "source_index": 1,
                },
            )
        ]

        chunker = Chunking(documents, self.embedding_model)
        chunks = chunker.semantic_chunking()

        vectorizer = Vectorizer(
            self.chroma_dir,
            self.embedding_model,
            collection_name="user_documents",
        )
        vector_store = vectorizer.build_vector(chunks)

        retriever_obj = vectorizer.create_retriever(vector_store)
        self.document_retriever = Retriever(retriever_obj)
        print(f"User document ingested (OCR text): {len(chunks)} chunks")
        return len(chunks)

    # ---------------- get the right retriever ----------------
    def _get_knowledge_retriever(self):
        if self.knowledge_retriever is None:
            vectorizer = Vectorizer(
                self.chroma_dir,
                self.embedding_model,
                collection_name="legal_knowledge",
            )
            # build it if the CUAD collection was never created
            if not Path(CHROMA_DIR).exists():
                self.ingest_cuad()
            else:
                vector_store = vectorizer.load_vector_store()
                retriever_obj = vectorizer.create_retriever(vector_store)
                self.knowledge_retriever = Retriever(retriever_obj)
        return self.knowledge_retriever

    def _get_document_retriever(self):
        if self.document_retriever is None:
            vectorizer = Vectorizer(
                self.chroma_dir,
                self.embedding_model,
                collection_name="user_documents",
            )
            vector_store = vectorizer.load_vector_store()
            retriever_obj = vectorizer.create_retriever(vector_store)
            self.document_retriever = Retriever(retriever_obj)
        return self.document_retriever

    # ---------------- get context (3 use cases) ----------------
    def get_context(self, question: str, has_document: bool, general: bool = False):
        """
        Returns (mode, rag_context)
          - has_document=False               -> CUAD          (Use case 1)
          - has_document=True, general=False -> uploaded doc  (Use case 2)
          - has_document=True, general=True  -> CUAD          (Use case 3)
        """
        if has_document and not general:
            retriever = self._get_document_retriever()
            context = retriever.search_documents(question)
            # fall back to CUAD if the document has nothing relevant
            if not context.strip():
                retriever = self._get_knowledge_retriever()
                context = retriever.search_documents(question)
                return "knowledge", context
            return "document", context
        else:
            retriever = self._get_knowledge_retriever()
            context = retriever.search_documents(question)
            return "knowledge", context

    # ---------------- get the final answer ----------------
    def get_answer(self, question: str, has_document: bool = False, general: bool = False) -> dict:

        mode, context = self.get_context(question, has_document, general)

        prompt = self.generator.build_prompt(
            question,
            self.memory.get_memory(),
            context,
            mode=mode,
        )

        answer = self.generator.generate(prompt)

        self.memory.add_memory(question, answer)

        return {"answer": answer, "mode": mode}

    def clear_memory(self):
        self.memory.clear()

 