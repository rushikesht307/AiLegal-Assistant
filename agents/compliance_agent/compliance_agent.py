from agents.base_agent import BaseLegalAgent
from agents.compliance_agent.compliance_rules import COMPLIANCE_CHECKLIST


class ComplianceAgent(BaseLegalAgent):
    AGENT_NAME = "Compliance Agent"
    DEFAULT_QUERY = "compliance governing law data protection"
    SYSTEM_PROMPT = (
        "You are a legal compliance assistant. Using only the context, check the "
        "document against this checklist and mark each item Present or Missing: "
        + "; ".join(COMPLIANCE_CHECKLIST) + ". "
        "Then give an overall status: Compliant, Partially Compliant, or Non-Compliant."
    )
