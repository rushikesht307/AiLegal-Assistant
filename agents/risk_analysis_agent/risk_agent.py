"""
Risk Analysis Agent   (Owner: Tejas Dighe)   [Day 4]
Flags risky clauses, missing protections, and gives a risk score using RAG + Gemini.
"""


class RiskAnalysisAgent:

    def __init__(self, knowledge_retriever, document_retriever, generator, memory, router):
        self.knowledge_retriever = knowledge_retriever
        self.document_retriever = document_retriever
        self.generator = generator
        self.memory = memory
        self.router = router

    def _get_context(self, question, has_document):
        if has_document and self.document_retriever is not None:
            ctx = self.document_retriever.search_documents(question or "risk liability termination indemnity")
            if ctx and ctx.strip():
                return "document", ctx
        ctx = self.knowledge_retriever.search_documents(question or "risky clauses")
        return "knowledge", ctx

    def run(self, question, has_document=False, general=False):
        mode, context = self._get_context(question, has_document)

        prompt = f"""You are a legal risk analysis assistant. Analyse the context below and:
1. Identify risky clauses (e.g., unlimited liability, one-sided terms, auto-renewal).
2. Point out any important MISSING protections (e.g., no confidentiality, no governing law).
3. Give an overall risk level: Low, Medium, or High, with a one-line reason.
Only use the context. Do not invent clauses.

Context:
{context}

Risk analysis:"""

        answer = self.generator.generate(prompt)
        self.memory.add_memory(question, answer)
        return {"agent": "Risk Analysis Agent", "answer": answer, "mode": mode}
