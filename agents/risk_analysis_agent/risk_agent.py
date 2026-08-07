"""
Risk Analysis Agent (LangChain)   (Owner: Tejas Dighe)
Flags risky / missing clauses and gives an overall risk level.
"""

from agents.base_agent import BaseLegalAgent


class RiskAnalysisAgent(BaseLegalAgent):
    AGENT_NAME = "Risk Analysis Agent"
    DEFAULT_QUERY = "risk liability termination indemnity"
    SYSTEM_PROMPT = (
        "You are a legal risk analysis assistant. From the context: "
        "1) identify risky clauses (unlimited liability, one-sided terms, auto-renewal); "
        "2) point out important MISSING protections (no confidentiality, no governing law); "
        "3) give an overall risk level (Low, Medium, High) with a one-line reason. "
        "Only use the context; do not invent clauses."
    )
