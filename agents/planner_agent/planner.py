# """
# Planner (Supervisor) Agent - LangGraph, one node per agent.
# Registry holds routing keywords; nodes are written explicitly.

# Flow:  START -> guardrail -> supervisor -> [agent node] -> END
# """

# from typing import TypedDict
# from langgraph.graph import StateGraph, START, END

# from guardrails.guardrails import validate_input, is_off_topic, add_disclaimer

# from agents.legal_rag_agent.qa_agent import QAAgent
# from agents.clause_extraction_agent.clause_extractor import ClauseExtractionAgent
# from agents.risk_analysis_agent.risk_agent import RiskAnalysisAgent
# from agents.compliance_agent.compliance_agent import ComplianceAgent
# from agents.contract_comparison_agent.comparison_agent import ContractComparisonAgent
# from agents.obligation_agent.obligation_agent import ObligationAgent
# from agents.report_generator_agent.report_generator import ReportGeneratorAgent


# # ---- registry: key -> routing keywords ----
# AGENT_REGISTRY = {
#     "report":     ["report", "full analysis", "generate report"],
#     "risk":       ["risk", "risky", "red flag", "danger"],
#     "compliance": ["compliance", "compliant", "regulation", "gdpr"],
#     "comparison": ["compare", "difference", "version", "vs "],
#     "obligation": ["deadline", "obligation", "renew", "due date", "notice period"],
#     "clause":     ["clause", "extract", "termination", "confidential", "liability"],
#     "qa":         [],   # default fallback
# }


# def pick_agent(question):
#     """Use the registry keywords to decide which agent should handle it."""
#     q = question.lower()
#     for key, keywords in AGENT_REGISTRY.items():
#         if any(kw in q for kw in keywords):
#             return key
#     return "qa"


# class State(TypedDict):
#     question: str
#     has_document: bool
#     general: bool
#     route: str
#     agent: str
#     answer: str
#     mode: str
#     blocked: bool


# class Planner:

#     def __init__(self, pipeline):
#         self.pipeline = pipeline
#         self.graph = self._build_graph()

#     # ---- helper: build an agent with the shared components ----
#     def _build(self, agent_cls):
#         self.pipeline._get_knowledge_retriever()
#         return agent_cls(
#             self.pipeline.knowledge_retriever,
#             self.pipeline.document_retriever,
#             self.pipeline.generator,
#             self.pipeline.memory,
#             self.pipeline.router,
#         )

#     # ---- helper: run an agent and fill the state ----
#     def _run(self, agent, state):
#         result = agent.run(state["question"], state["has_document"], state["general"])
#         state["agent"] = result["agent"]
#         state["answer"] = add_disclaimer(result["answer"])
#         state["mode"] = result["mode"]
#         return state

#     # ================= NODES =================
#     def guardrail(self, state):
#         ok, msg = validate_input(state["question"])
#         if not ok or is_off_topic(state["question"]):
#             state["blocked"] = True
#             state["agent"] = "Guardrail"
#             state["answer"] = msg if not ok else "I can only help with legal questions."
#             state["mode"] = "blocked"
#         else:
#             state["blocked"] = False
#         return state

#     def supervisor(self, state):
#         state["route"] = pick_agent(state["question"])
#         return state

#     # ---- one explicit node per agent ----
#     def qa_node(self, state):
#         return self._run(self._build(QAAgent), state)

#     def clause_node(self, state):
#         return self._run(self._build(ClauseExtractionAgent), state)

#     def risk_node(self, state):
#         return self._run(self._build(RiskAnalysisAgent), state)

#     def compliance_node(self, state):
#         return self._run(self._build(ComplianceAgent), state)

#     def comparison_node(self, state):
#         return self._run(self._build(ContractComparisonAgent), state)

#     def obligation_node(self, state):
#         return self._run(self._build(ObligationAgent), state)

#     def report_node(self, state):
#         return self._run(self._build(ReportGeneratorAgent), state)

#     # ================= EDGE DECISION FUNCTIONS =================
#     def after_guardrail(self, state):
#         if state["blocked"]:
#             return "end"
#         return "supervisor"

#     def after_supervisor(self, state):
#         return state["route"]

#     # ================= BUILD THE GRAPH =================
#     def _build_graph(self):
#         g = StateGraph(State)

#         g.add_node("guardrail", self.guardrail)
#         g.add_node("supervisor", self.supervisor)
#         g.add_node("qa", self.qa_node)
#         g.add_node("clause", self.clause_node)
#         g.add_node("risk", self.risk_node)
#         g.add_node("compliance", self.compliance_node)
#         g.add_node("comparison", self.comparison_node)
#         g.add_node("obligation", self.obligation_node)
#         g.add_node("report", self.report_node)

#         # START -> guardrail
#         g.add_edge(START, "guardrail")

#         # guardrail -> supervisor or END
#         g.add_conditional_edges(
#             "guardrail",
#             self.after_guardrail,
#             {"end": END, "supervisor": "supervisor"},
#         )

#         # supervisor -> the chosen agent node
#         g.add_conditional_edges(
#             "supervisor",
#             self.after_supervisor,
#             {
#                 "qa": "qa",
#                 "clause": "clause",
#                 "risk": "risk",
#                 "compliance": "compliance",
#                 "comparison": "comparison",
#                 "obligation": "obligation",
#                 "report": "report",
#             },
#         )

#         # every agent node -> END
#         g.add_edge("qa", END)
#         g.add_edge("clause", END)
#         g.add_edge("risk", END)
#         g.add_edge("compliance", END)
#         g.add_edge("comparison", END)
#         g.add_edge("obligation", END)
#         g.add_edge("report", END)

#         return g.compile()

#     # ================= ENTRY =================
#     def route(self, question, has_document=False, general=False):
#         if has_document:
#             self.pipeline._get_document_retriever()
#         state = {
#             "question": question, "has_document": has_document, "general": general,
#             "route": "", "agent": "", "answer": "", "mode": "", "blocked": False,
#         }
#         out = self.graph.invoke(state)
#         return {"agent": out["agent"], "answer": out["answer"], "mode": out["mode"]}
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
        "keywords": ["report", "full analysis", "generate report", "summary", "summarize"],
        "description": (
            "Generate comprehensive legal reports, executive summaries, "
            "contract reviews, findings, recommendations, and conclusions."
        ),
    },
    "risk": {
        "keywords": ["risk", "risky", "red flag", "danger", "liability", "exposure"],
        "description": (
            "Identify contractual risks, liabilities, legal exposure, "
            "red flags, risks to parties, and risky provisions."
        ),
    },
    "compliance": {
        "keywords": ["compliance", "compliant", "regulation", "gdpr", "policy", "standard"],
        "description": (
            "Analyze regulatory compliance, GDPR compliance, policy compliance, "
            "industry standards, and legal obligations."
        ),
    },
    "comparison": {
        "keywords": ["compare", "difference", "version", "vs ", "changes", "modification"],
        "description": (
            "Compare contracts, compare document versions, identify changes, "
            "differences, additions, removals, and modifications."
        ),
    },
    "obligation": {
        "keywords": ["deadline", "obligation", "renew", "due date", "notice period",
                     "milestone", "deliverable"],
        "description": (
            "Extract obligations, deliverables, deadlines, milestones, "
            "renewals, notice periods, and action items."
        ),
    },
    "clause": {
        "keywords": ["clause", "extract", "termination", "confidential", "indemnity",
                     "governing law", "arbitration", "force majeure", "payment"],
        "description": (
            "Extract or analyze clauses including termination, indemnity, "
            "confidentiality, liability, governing law, arbitration, "
            "force majeure, and payment clauses."
        ),
    },
    "qa": {
        "keywords": [],   # default fallback
        "description": (
            "Answer general legal questions, legal reasoning queries, "
            "and legal knowledge requests."
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