from agents.clause_extraction_agent.clause_types import CLAUSE_TYPES


class ClauseExtractionAgent:

    def __init__(self, knowledge_retriever, document_retriever, generator, memory, router):
        self.knowledge_retriever = knowledge_retriever
        self.document_retriever = document_retriever
        self.generator = generator
        self.memory = memory
        self.router = router

    def _get_context(self, question, has_document):
        # clauses are extracted from the DOCUMENT (fall back to CUAD if no doc)
        if has_document and self.document_retriever is not None:
            ctx = self.document_retriever.search_documents(question or "clauses terms conditions")
            if ctx and ctx.strip():
                return "document", ctx
        ctx = self.knowledge_retriever.search_documents(question or "standard clauses")
        return "knowledge", ctx

    def run(self, question, has_document=False, general=False):
        mode, context = self._get_context(question, has_document)

        clause_list = ", ".join(CLAUSE_TYPES)
        prompt = f"""You are a legal clause extraction assistant. From the context below,
identify and extract the key legal clauses. Focus on these clause types: {clause_list}.

For each clause you find, give:
  - the clause name
  - a short summary of what it says
Only use the context. If a clause is not present, do not invent it.

Context:
{context}

List the clauses found:"""

        answer = self.generator.generate(prompt)
        self.memory.add_memory(question, answer)
        return {"agent": "Clause Extraction Agent", "answer": answer, "mode": mode}
