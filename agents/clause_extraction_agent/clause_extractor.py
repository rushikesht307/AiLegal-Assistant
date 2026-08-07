from agents.base_agent import BaseLegalAgent
from agents.clause_extraction_agent.clause_types import CLAUSE_TYPES


class ClauseExtractionAgent(BaseLegalAgent):
    AGENT_NAME = "Clause Extraction Agent"
    DEFAULT_QUERY = "clauses terms conditions"
    SYSTEM_PROMPT = (
        "You are a legal clause extraction assistant. From the context, identify and "
        "summarise the key clauses. Focus on: " + ", ".join(CLAUSE_TYPES) + ". "
        "For each clause found, give the clause name in bold and a short summary. "
        "Only use the context; do not invent clauses."
    )
