"""
Contract Comparison Agent (LangChain)   (Owner: Adarsh Singh)
Compares the uploaded document against the CUAD standard reference.
"""

from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseLegalAgent


class ContractComparisonAgent(BaseLegalAgent):
    AGENT_NAME = "Contract Comparison Agent"

    def run(self, question, has_document=False, general=False):
        if not has_document or self.document_retriever is None:
            return {"agent": self.AGENT_NAME,
                    "answer": "Please upload a document first. To compare two versions, upload the second version too.",
                    "mode": "knowledge"}

        doc_ctx = self.document_retriever.search_documents(question or "key clauses terms")
        std_ctx = self.knowledge_retriever.search_documents(question or "standard contract clauses")

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a contract comparison assistant. Compare the user's document against the "
             "standard reference. Point out: 1) clauses present in the document; 2) clauses that "
             "differ from the standard; 3) standard clauses that are MISSING. Use only the contexts."),
            ("human",
             "User Document Context:\n{doc}\n\nStandard Reference Context:\n{std}\n\nComparison:"),
        ])
        messages = prompt.format_messages(doc=doc_ctx, std=std_ctx)
        response = self.generator.model.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = " ".join((c.get("text","") if isinstance(c, dict) else str(c)) for c in content).strip()
        self.memory.add_memory(question, content)
        return {"agent": self.AGENT_NAME, "answer": content, "mode": "document"}
