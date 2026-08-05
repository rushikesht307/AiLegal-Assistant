from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


class Chunking:
    def __init__(self, documents, embedding_model):
        self.documents = documents
        self.embedding_model = embedding_model

    def semantic_chunking(self):

        # Create the semantic chunker
        chunker = SemanticChunker(self.embedding_model)
        chunk = chunker.split_documents(self.documents)
        print("Semantic chunks created")
        return chunk
