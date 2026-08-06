class ContractComparisonAgent:

    def __init__(self, knowledge_retriever, document_retriever, generator, memory, router):
        self.knowledge_retriever = knowledge_retriever
        self.document_retriever = document_retriever
        self.generator = generator
        self.memory = memory
        self.router = router

    def run(self, question, has_document=False, general=False):
        if not has_document or self.document_retriever is None:
            return {"agent": "Contract Comparison Agent",
                    "answer": "Please upload a document first. To compare two versions, upload the second version too.",
                    "mode": "knowledge"}

        doc_ctx = self.document_retriever.search_documents(question or "key clauses terms")
        std_ctx = self.knowledge_retriever.search_documents(question or "standard contract clauses")

        prompt = f"""You are a contract comparison assistant. Compare the user's document against the
standard (reference) clauses. Point out:
1. Clauses present in the document.
2. Clauses that differ from the standard.
3. Standard clauses that are MISSING from the document.
Only use the two contexts below.

User Document Context:
{doc_ctx}

Standard (Reference) Context:
{std_ctx}

Comparison:"""

        answer = self.generator.generate(prompt)
        self.memory.add_memory(question, answer)
        return {"agent": "Contract Comparison Agent", "answer": answer, "mode": "document"}
