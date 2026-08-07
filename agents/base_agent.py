"""
BaseLegalAgent   (LangChain)
All specialized agents inherit from this. It uses LangChain's ChatPromptTemplate
+ the ChatGoogleGenerativeAI model (Gemini) to answer, grounded in retrieved
context. Each agent only sets its own SYSTEM_PROMPT and TASK.

Shared components are passed in (same style as qa_agent.py):
  knowledge_retriever, document_retriever, generator, memory, router
'"'"'generator'"'"' is your Generator wrapper (holds the LangChain Gemini model).
"""

from langchain_core.prompts import ChatPromptTemplate


class BaseLegalAgent:

    AGENT_NAME = "Base Agent"
    SYSTEM_PROMPT = "You are a helpful legal assistant. Answer only from the context."
    DEFAULT_QUERY = "legal document"

    def __init__(self, knowledge_retriever, document_retriever, generator, memory, router):
        self.knowledge_retriever = knowledge_retriever
        self.document_retriever = document_retriever
        self.generator = generator
        self.memory = memory
        self.router = router

    # ---- pick document vs CUAD (uses the shared router) ----
    def get_context(self, question, has_document, general=False):
        query = question or self.DEFAULT_QUERY
        if has_document and not general and self.document_retriever is not None:
            ctx = self.document_retriever.search_documents(query)
            if ctx and ctx.strip():
                return "document", ctx
            ctx = self.knowledge_retriever.search_documents(query)
            return "knowledge", ctx
        ctx = self.knowledge_retriever.search_documents(query)
        return "knowledge", ctx

    # ---- build a LangChain prompt and run the LLM ----
    def _ask_llm(self, question, context):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human",
             "Context:\n{context}\n\n"
             "Conversation so far:\n{memory}\n\n"
             "Question: {question}\n\n"
             "Answer using ONLY the context. If the answer is not in the context, "
             "say you do not have enough information."),
        ])
        messages = prompt.format_messages(
            context=context,
            memory=self.memory.get_memory(),
            question=question,
        )
        # generator.model is the LangChain ChatGoogleGenerativeAI
        response = self.generator.model.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = " ".join(
                (c.get("text", "") if isinstance(c, dict) else str(c)) for c in content
            ).strip()
        return content

    def run(self, question, has_document=False, general=False):
        mode, context = self.get_context(question, has_document, general)
        answer = self._ask_llm(question, context)
        self.memory.add_memory(question, answer)
        return {"agent": self.AGENT_NAME, "answer": answer, "mode": mode}
