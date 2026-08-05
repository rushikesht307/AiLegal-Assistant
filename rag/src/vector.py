from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path


class Vectorizer:
    """
    Handles the vector store. Uses a 'collection_name' so we can keep TWO
    separate stores in the project:
        - "legal_knowledge" -> the CUAD knowledge base
        - "user_documents"  -> the document the user uploaded
    Both are persisted under the same CHROMA_DIR but in different collections.
    """

    def __init__(self, CHROMA_DIR, embedding, collection_name="legal_knowledge"):
        self.CHROMA_DIR = CHROMA_DIR
        self.embedding = embedding
        self.collection_name = collection_name

    def load_vector_store(self):
        vector_db = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.CHROMA_DIR),
            embedding_function=self.embedding,
        )
        return vector_db

    def build_vector(self, chunks=None):
        vector_db = Chroma.from_documents(
            documents=chunks,
            collection_name=self.collection_name,
            persist_directory=str(self.CHROMA_DIR),
            embedding=self.embedding,
        )
        return vector_db

    def create_retriever(self, vector_store, k=3):
        return vector_store.as_retriever(
            search_kwargs={"k": k}
        )
