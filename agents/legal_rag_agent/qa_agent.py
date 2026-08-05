
class QAAgent:

    def __init__(self, knowledge_retriever, document_retriever, generator, memory, router):
        self.knowledge_retriever = knowledge_retriever
        self.document_retriever = document_retriever
        self.generator = generator
        self.memory = memory
        self.router = router

    def get_context(self, question, has_document, general=False):
        # the router decides: "document" or "knowledge"
        mode = self.router.route(
            question,
            has_document=has_document,
            document_retriever=self.document_retriever,
            general_hint=general,
        )

        if mode == "document" and self.document_retriever is not None:
            context = self.document_retriever.search_documents(question)
            if not context or not context.strip():
                context = self.knowledge_retriever.search_documents(question)
                return "knowledge", context
            return "document", context

        context = self.knowledge_retriever.search_documents(question)
        return "knowledge", context

    def answer(self, question, has_document=False, general=False) -> dict:
        # 1. retrieve (router picks the source)
        mode, context = self.get_context(question, has_document, general)

        # 2. build the grounded prompt (Gemini)
        prompt = self.generator.build_prompt(
            question, self.memory.get_memory(), context, mode=mode
        )

        # 3. generate the answer
        answer_text = self.generator.generate(prompt)

        # 4. remember it
        self.memory.add_memory(question, answer_text)

        return {"answer": answer_text, "mode": mode}

    def clear_memory(self):
        self.memory.clear()