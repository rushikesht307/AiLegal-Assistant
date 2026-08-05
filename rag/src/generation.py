from langchain_google_genai import (
    ChatGoogleGenerativeAI
)


class Generator:

    def __init__(self, model_name: str, api_key: str,):

        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=api_key,
        )

    def generate(self, prompt: str,) -> str:

        response = self.model.invoke(prompt)
        content = response.content

        # flash-lite / newer Gemini can return a list of parts, not a plain string
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            return " ".join(p for p in parts if p).strip()

        return str(content)

    def build_prompt(self, question: str, memory_context: str, rag_context: str, mode: str = "document") -> str:
        """
        mode:
          - "document"  -> answering from the user's uploaded document
          - "knowledge" -> answering from the CUAD legal knowledge base
                           (no document uploaded, OR a general legal question)
        """

        source_line = (
            "the uploaded legal document"
            if mode == "document"
            else "the CUAD legal knowledge base (general legal reference)"
        )

        return f"""You are a helpful legal assistant that answers questions strictly using the information provided below. You must not use any outside knowledge or make assumptions beyond what is given. Your answers are based on {source_line}.

Conversation History:
{memory_context}

Retrieved Context:
{rag_context}

Current Question:
{question}

Instructions:
1. If the Current Question is about the conversation itself (e.g., "what did I ask before", "repeat my last question", "summarize our chat"), answer it directly using the Conversation History above — do not require Retrieved Context for these meta-questions.
2. For all other (legal/factual) questions, answer only using facts found in the Retrieved Context. Use Conversation History only to resolve references (e.g., "it", "that", follow-ups) — not as a source of new legal facts.
3. If the Retrieved Context does not contain enough information to answer a legal question confidently, respond exactly with: "I do not have enough information in the provided documents to answer this."
4. Do not guess, infer beyond the text, or use prior knowledge about the topic.
5. If multiple sources in the context disagree, mention the discrepancy instead of picking one silently.
6. Provide a detailed and well-explained answer of at least 3-5 sentences when the context supports it. Explain the relevant points clearly and give a complete, helpful response. Avoid one-line answers.
7. If the context includes source names/file names, cite them briefly at the end, e.g., (Source: filename).

Answer:"""