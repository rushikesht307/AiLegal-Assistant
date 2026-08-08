"""
BaseLegalAgent (LangChain)

All specialized agents inherit from this.

Shared components:
- knowledge_retriever
- document_retriever
- generator
- memory
- router
"""

from langchain_core.prompts import ChatPromptTemplate


class BaseLegalAgent:

    AGENT_NAME = "Base Agent"

    SYSTEM_PROMPT = (
        "You are a helpful legal assistant. "
        "Answer only from the provided context."
    )

    DEFAULT_QUERY = "legal document"

    def __init__(
        self,
        knowledge_retriever,
        document_retriever,
        generator,
        memory,
        router,
    ):
        self.knowledge_retriever = knowledge_retriever
        self.document_retriever = document_retriever
        self.generator = generator
        self.memory = memory
        self.router = router

    # =====================================================
    # Context Selection
    # =====================================================

    def get_context(self, question, has_document, general=False):

        query = question or self.DEFAULT_QUERY

        # No uploaded document available
        if not has_document or self.document_retriever is None:

            ctx = self.knowledge_retriever.search_documents(query)

            if isinstance(ctx, list):
                ctx = "\n\n".join(
                    doc.page_content
                    if hasattr(doc, "page_content")
                    else str(doc)
                    for doc in ctx
                )

            return "knowledge", ctx

        router_prompt = f"""
        You are a legal routing agent.
        Determine where the answer should come from.
        Options:
        knowledge
        - General legal concepts
        - Legal definitions
        - NDA explanations
        - Contract law
        - Arbitration
        - GDPR
        - Legal principles
        document
        - Questions about the uploaded contract
        - Clause extraction
        - Summarization
        - Obligations in the uploaded file
        - Information found in the uploaded document
        Examples:
        Question: What is an NDA?
        Answer: knowledge
        Question: Explain arbitration.
        Answer: knowledge
        Question: What clauses are usually present in an NDA?
        Answer: knowledge
        Question: Summarize the uploaded agreement.
        Answer: document
        Question: What does the termination clause say?
        Answer: document
        Question: Extract obligations from the contract.
        Answer: document
        Question:
        {question}

        Return ONLY:
        knowledge
        or
        document
        """
        try:
            response = self.generator.model.invoke(router_prompt)
            content = response.content
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "")
                    if isinstance(item, dict)
                    else str(item)
                    for item in content
                )

            decision = str(content).strip().lower()

        except Exception:
            decision = "knowledge"

        print(f"[ROUTER] Question: {question}")
        print(f"[ROUTER] Decision: {decision}")

        # ---------------------------------------------
        # Use Document Retriever
        # ---------------------------------------------
        if decision == "document":

            try:
                ctx = self.document_retriever.search_documents(query)
                if isinstance(ctx, list):
                    ctx = "\n\n".join(
                        doc.page_content
                        if hasattr(doc, "page_content")
                        else str(doc)
                        for doc in ctx
                    )

                if ctx and str(ctx).strip():
                    return "document", ctx

            except Exception as e:
                print(f"[DOCUMENT RETRIEVER ERROR] {e}")

        # ---------------------------------------------
        # Use Knowledge Retriever
        # ---------------------------------------------
        try:

            ctx = self.knowledge_retriever.search_documents(query)
            if isinstance(ctx, list):
                ctx = "\n\n".join(
                    doc.page_content
                    if hasattr(doc, "page_content")
                    else str(doc)
                    for doc in ctx
                )

            return "knowledge", ctx

        except Exception as e:

            print(f"[KNOWLEDGE RETRIEVER ERROR] {e}")

            return "knowledge", ""

    # =====================================================
    # LLM Call
    # =====================================================

    def _ask_llm(self, question, context):

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                (
                    "human",
                    "Context:\n{context}\n\n"
                    "Conversation so far:\n{memory}\n\n"
                    "Question: {question}\n\n"
                    "Answer ONLY using the supplied context. "
                    "If the answer is not available in the context, "
                    "say you do not have enough information."
                ),
            ]
        )

        messages = prompt.format_messages(
            context=context,
            memory=self.memory.get_memory(),
            question=question,
        )

        response = self.generator.model.invoke(messages)

        content = response.content

        if isinstance(content, list):
            content = " ".join(
                item.get("text", "")
                if isinstance(item, dict)
                else str(item)
                for item in content
            ).strip()

        return str(content)

    # =====================================================
    # Main Entry
    # =====================================================

    def run(self, question, has_document=False, general=False):

        mode, context = self.get_context(
            question,
            has_document,
            general,
        )

        answer = self._ask_llm(
            question,
            context,
        )

        self.memory.add_memory(
            question,
            answer,
        )

        return {
            "agent": self.AGENT_NAME,
            "answer": answer,
            "mode": mode,
        }