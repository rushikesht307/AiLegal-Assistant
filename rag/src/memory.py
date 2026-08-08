class Memory:
    def __init__(self):
        self.messages = []

    def add_memory(
        self,
        question: str,
        answer: str,
    ):
        self.messages.append(
            {
                "question": question,
                "answer": answer,
            }
        )

        self.messages = self.messages[-5:]
    def get_memory(self) -> str:
        if not self.messages:
            return ""
        context = []
        for item in self.messages:
            context.append(
                f"User: {item['question']}"
            )

            context.append(
                f"Assistant: {item['answer']}"
            )

        return "\n".join(context)

    def clear(self):
        self.messages.clear()
