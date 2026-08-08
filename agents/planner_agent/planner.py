"""
Planner (Supervisor) Agent - LangGraph
Hybrid routing: keyword match first (fast), then LLM router (smart fallback).

Flow:  START -> guardrail -> supervisor -> [agent node] -> END
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from guardrails.guardrails import (
    validate_input,
    is_off_topic,
    add_disclaimer,
)
from agents.legal_rag_agent.qa_agent import QAAgent
from agents.clause_extraction_agent.clause_extractor import ClauseExtractionAgent
from agents.risk_analysis_agent.risk_agent import RiskAnalysisAgent
from agents.compliance_agent.compliance_agent import ComplianceAgent
from agents.contract_comparison_agent.comparison_agent import ContractComparisonAgent
from agents.obligation_agent.obligation_agent import ObligationAgent
from agents.report_generator_agent.report_generator import ReportGeneratorAgent


# ==========================================================
# Agent Registry
#   keywords     -> fast, reliable first check
#   description  -> used by the LLM router as a fallback
# ==========================================================
AGENT_REGISTRY = {
    "report": {
        "keywords": [
            "generate report",
            "create report",
            "download report",
            "export report",
            "full legal report",
            "pdf report",
            "risk report",
            "executive report"
        ],
        "description": (
            "Generate a downloadable PDF legal report. "
            "Use ONLY when the user explicitly asks to "
            "generate, create, export, or download a report."
        ),
    },

    "risk": {
        "keywords": [
            "risk",
            "risky",
            "red flag",
            "danger",
            "liability",
            "exposure"
        ],
        "description": (
            "Identify contractual risks, liabilities, red flags, "
            "legal exposure and risky provisions."
        ),
    },

    "compliance": {
        "keywords": [
            "compliance",
            "compliant",
            "regulation",
            "gdpr",
            "policy",
            "standard"
        ],
        "description": (
            "Analyze regulatory compliance and legal requirements."
        ),
    },

    "comparison": {
        "keywords": [
            "compare",
            "difference",
            "version",
            "changes",
            "compare contracts"
        ],
        "description": (
            "Compare two contracts or versions."
        ),
    },

    "obligation": {
        "keywords": [
            "obligation",
            "deadline",
            "due date",
            "renewal",
            "deliverable",
            "notice period"
        ],
        "description": (
            "Extract obligations, action items and deadlines."
        ),
    },

    "clause": {
        "keywords": [
            "clause",
            "termination clause",
            "confidentiality clause",
            "indemnity clause",
            "arbitration clause",
            "governing law clause",
            "extract clause"
        ],
        "description": (
            "Extract or analyze contract clauses."
        ),
    },

    "qa": {
        "keywords": [],
        "description": (
            "Answer general legal questions, explain documents, "
            "summarize agreements, answer questions about uploaded "
            "documents, and provide legal explanations."
        ),
    },
}


# ==========================================================
# LangGraph State
# ==========================================================
class State(TypedDict):
    question: str
    has_document: bool
    general: bool
    route: str
    agent: str
    answer: str
    mode: str
    blocked: bool


# ==========================================================
# Planner
# ==========================================================
class Planner:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.graph = self._build_graph()

    # ------------------------------------------------------
    # Shared agent builder
    # ------------------------------------------------------
    def _build(self, agent_cls):
        self.pipeline._get_knowledge_retriever()
        return agent_cls(
            self.pipeline.knowledge_retriever,
            self.pipeline.document_retriever,
            self.pipeline.generator,
            self.pipeline.memory,
            self.pipeline.router,
        )

    # ------------------------------------------------------
    # Shared agent runner
    # ------------------------------------------------------
    def _run(self, agent, state):
        result = agent.run(
            state["question"],
            state["has_document"],
            state["general"],
        )
        state["agent"] = result["agent"]
        state["answer"] = add_disclaimer(result["answer"])
        state["mode"] = result["mode"]
        return state

    # ------------------------------------------------------
    # LAYER 1 - keyword routing (fast)
    # ------------------------------------------------------
    def _keyword_route(self, question):
        q = question.lower()
        for key, cfg in AGENT_REGISTRY.items():
            if any(kw in q for kw in cfg["keywords"]):
                return key
        return None      # no keyword matched

    # ------------------------------------------------------
    # LAYER 2 - LLM routing (smart fallback)
    # ------------------------------------------------------
    def _llm_route(self, question):
        agents_text = "\n".join(
            f"- {name}: {cfg['description']}" for name, cfg in AGENT_REGISTRY.items()
        )
        valid_keys = ", ".join(AGENT_REGISTRY.keys())

        prompt = (
            "You are a Legal AI Supervisor. Choose the SINGLE best agent for the "
            "user's question.\n\n"
            f"Available agents:\n{agents_text}\n\n"
            f"User question: {question}\n\n"
            f"Reply with ONLY one agent name from: {valid_keys}\n"
            "Answer:"
        )

        try:
            response = self.pipeline.generator.model.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                content = " ".join(
                    (c.get("text", "") if isinstance(c, dict) else str(c))
                    for c in content
                )
            text = content.strip().lower()
            for key in AGENT_REGISTRY:
                if key in text:
                    return key
        except Exception as e:
            print(f"[SUPERVISOR] LLM error: {e}")
        return "qa"

    # ------------------------------------------------------
    # Guardrail node
    # ------------------------------------------------------
    def guardrail(self, state):
        ok, msg = validate_input(state["question"])
        if not ok or is_off_topic(state["question"]):
            state["blocked"] = True
            state["agent"] = "Guardrail"
            state["answer"] = msg if not ok else "I can only help with legal questions."
            state["mode"] = "blocked"
        else:
            state["blocked"] = False
        return state

    # ------------------------------------------------------
    # Supervisor node (HYBRID: keywords first, then LLM)
    # ------------------------------------------------------
    def supervisor(self, state):
        question = state["question"]

        # LAYER 1: try keywords (fast)
        route = self._keyword_route(question)

        if route:
            print(f"[SUPERVISOR] keyword match -> '{route}'  ('{question}')")
        else:
            # LAYER 2: no keyword matched -> ask the LLM
            route = self._llm_route(question)
            print(f"[SUPERVISOR] LLM route -> '{route}'  ('{question}')")

        state["route"] = route
        return state

    # ------------------------------------------------------
    # Agent nodes
    # ------------------------------------------------------
    def qa_node(self, state):
        return self._run(self._build(QAAgent), state)

    def clause_node(self, state):
        return self._run(self._build(ClauseExtractionAgent), state)

    def risk_node(self, state):
        return self._run(self._build(RiskAnalysisAgent), state)

    def compliance_node(self, state):
        return self._run(self._build(ComplianceAgent), state)

    def comparison_node(self, state):
        return self._run(self._build(ContractComparisonAgent), state)

    def obligation_node(self, state):
        return self._run(self._build(ObligationAgent), state)

    def report_node(self, state):
        return self._run(self._build(ReportGeneratorAgent), state)

    # ------------------------------------------------------
    # Edge decision functions
    # ------------------------------------------------------
    def after_guardrail(self, state):
        if state["blocked"]:
            return "end"
        return "supervisor"

    def after_supervisor(self, state):
        return state["route"]

    # ------------------------------------------------------
    # Build the graph
    # ------------------------------------------------------
    def _build_graph(self):
        g = StateGraph(State)

        g.add_node("guardrail", self.guardrail)
        g.add_node("supervisor", self.supervisor)
        g.add_node("qa", self.qa_node)
        g.add_node("clause", self.clause_node)
        g.add_node("risk", self.risk_node)
        g.add_node("compliance", self.compliance_node)
        g.add_node("comparison", self.comparison_node)
        g.add_node("obligation", self.obligation_node)
        g.add_node("report", self.report_node)

        g.add_edge(START, "guardrail")

        g.add_conditional_edges(
            "guardrail",
            self.after_guardrail,
            {"end": END, "supervisor": "supervisor"},
        )

        g.add_conditional_edges(
            "supervisor",
            self.after_supervisor,
            {
                "qa": "qa",
                "clause": "clause",
                "risk": "risk",
                "compliance": "compliance",
                "comparison": "comparison",
                "obligation": "obligation",
                "report": "report",
            },
        )

        g.add_edge("qa", END)
        g.add_edge("clause", END)
        g.add_edge("risk", END)
        g.add_edge("compliance", END)
        g.add_edge("comparison", END)
        g.add_edge("obligation", END)
        g.add_edge("report", END)

        return g.compile()

    # ------------------------------------------------------
    # Entry function
    # ------------------------------------------------------
    def route(self, question, has_document=False, general=False):
        if has_document:
            self.pipeline._get_document_retriever()

        state = {
            "question": question,
            "has_document": has_document,
            "general": general,
            "route": "",
            "agent": "",
            "answer": "",
            "mode": "",
            "blocked": False,
        }
        output = self.graph.invoke(state)
        return {
            "agent": output["agent"],
            "answer": output["answer"],
            "mode": output["mode"],
        }