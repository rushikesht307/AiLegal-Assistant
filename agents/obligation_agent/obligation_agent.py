"""
Obligation & Deadline Agent   (Owner: Bikkini Vasanth Kumar)   [Day 5]
Extracts obligations, dates, notice periods, renewals and milestones.
"""


class ObligationAgent:

    def __init__(self, knowledge_retriever, document_retriever, generator, memory, router):
        self.knowledge_retriever = knowledge_retriever
        self.document_retriever = document_retriever
        self.generator = generator
        self.memory = memory
        self.router = router

    def _get_context(self, question, has_document):
        if has_document and self.document_retriever is not None:
            ctx = self.document_retriever.search_documents(question or "dates notice period deadline renewal obligation")
            if ctx and ctx.strip():
                return "document", ctx
        ctx = self.knowledge_retriever.search_documents(question or "obligations deadlines")
        return "knowledge", ctx

    def run(self, question, has_document=False, general=False):
        mode, context = self._get_context(question, has_document)

        prompt = f"""You are an obligation and deadline extraction assistant. From the context below,
extract all:
- Key dates (start, end, effective, expiry)
- Notice periods (e.g., 30 days notice)
- Renewal terms
- Payment due dates / milestones
- Any obligations each party must fulfil
Only use the context. Present them as a clear list.

Context:
{context}

Obligations & deadlines:"""

        answer = self.generator.generate(prompt)
        self.memory.add_memory(question, answer)
        return {"agent": "Obligation & Deadline Agent", "answer": answer, "mode": mode}
