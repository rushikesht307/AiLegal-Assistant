from agents.compliance_agent.compliance_rules import COMPLIANCE_CHECKLIST


class ComplianceAgent:

    def __init__(self, knowledge_retriever, document_retriever, generator, memory, router):
        self.knowledge_retriever = knowledge_retriever
        self.document_retriever = document_retriever
        self.generator = generator
        self.memory = memory
        self.router = router

    def _get_context(self, question, has_document):
        if has_document and self.document_retriever is not None:
            ctx = self.document_retriever.search_documents(question or "compliance policy governing law data protection")
            if ctx and ctx.strip():
                return "document", ctx
        ctx = self.knowledge_retriever.search_documents(question or "compliance requirements")
        return "knowledge", ctx

    def run(self, question, has_document=False, general=False):
        mode, context = self._get_context(question, has_document)

        checklist = "\n".join(f"- {item}" for item in COMPLIANCE_CHECKLIST)
        prompt = f"""You are a legal compliance assistant. Using ONLY the context below, check the
document against this compliance checklist and mark each item as Present or Missing:

{checklist}

Then give an overall status: Compliant, Partially Compliant, or Non-Compliant.

Context:
{context}

Compliance check:"""

        answer = self.generator.generate(prompt)
        self.memory.add_memory(question, answer)
        return {"agent": "Compliance Agent", "answer": answer, "mode": mode}
