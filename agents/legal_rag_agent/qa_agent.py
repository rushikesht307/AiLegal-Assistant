from agents.base_agent import BaseLegalAgent


class QAAgent(BaseLegalAgent):
    AGENT_NAME = "Legal Q&A Agent"
    DEFAULT_QUERY = "legal question"
    SYSTEM_PROMPT = (
        "You are a legal Q&A assistant. Give a clear, detailed and well-explained "
        "answer of a few sentences, using only the provided context. Cite the source "
        "at the end as (Source: uploaded document) or (Source: CUAD knowledge base). "
        "Do not use outside knowledge."
    )
