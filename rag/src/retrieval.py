# retrieval.py
class Retriever:
    def __init__(self, retriever):
        self.retriever = retriever

    def search_documents(self, question: str) -> str:
        documents = self.retriever.invoke(question)
        return self.format_context(documents)

    def format_context(self, documents: list) -> str:
        if not documents:
            return ""

        blocks = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source_file", "unknown")
            blocks.append(f"[{i}] (source: {source})\n{doc.page_content}")

        return "\n\n".join(blocks)
