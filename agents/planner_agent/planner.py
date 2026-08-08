from typing import TypedDict
from pydantic import BaseModel
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
from agents.contract_comparison_agent.comparison_agent import (
    ContractComparisonAgent,
)
from agents.obligation_agent.obligation_agent import ObligationAgent
from agents.report_generator_agent.report_generator import (
    ReportGeneratorAgent,
)


# ==========================================================
# Agent Registry
# ==========================================================

AGENT_REGISTRY = {
    "report": (
        "Generate comprehensive legal reports, executive summaries, "
        "contract reviews, findings, recommendations, and conclusions."
    ),

    "risk": (
        "Identify contractual risks, liabilities, legal exposure, "
        "red flags, risks to parties, and risky provisions."
    ),

    "compliance": (
        "Analyze regulatory compliance, GDPR compliance, policy compliance, "
        "industry standards, and legal obligations."
    ),

    "comparison": (
        "Compare contracts, compare document versions, identify changes, "
        "differences, additions, removals, and modifications."
    ),

    "obligation": (
        "Extract obligations, deliverables, deadlines, milestones, "
        "renewals, notice periods, and action items."
    ),

    "clause": (
        "Extract or analyze clauses including termination, indemnity, "
        "confidentiality, liability, governing law, arbitration, "
        "force majeure, and payment clauses."
    ),

    "qa": (
        "Answer general legal questions, legal reasoning queries, "
        "and legal knowledge requests."
    ),
}


# ==========================================================
# Structured Output
# ==========================================================

class RouteDecision(BaseModel):
    agent: str


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
        self.router_llm = (
            self.pipeline.generator.model
            .with_structured_output(RouteDecision)
        )
        self.graph = self._build_graph()

    # ======================================================
    # Shared Agent Builder
    # ======================================================

    def _build(self, agent_cls):
        self.pipeline._get_knowledge_retriever()
        return agent_cls(
            self.pipeline.knowledge_retriever,
            self.pipeline.document_retriever,
            self.pipeline.generator,
            self.pipeline.memory,
            self.pipeline.router,
        )

    # ======================================================
    # Shared Agent Runner
    # ======================================================

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

    # ======================================================
    # Guardrail Node
    # ======================================================

    def guardrail(self, state):
        ok, msg = validate_input(
            state["question"]
        )
        if not ok or is_off_topic(
            state["question"]
        ):

            state["blocked"] = True
            state["agent"] = "Guardrail"

            state["answer"] = (
                msg
                if not ok
                else "I can only help with legal questions."
            )

            state["mode"] = "blocked"

        else:
            state["blocked"] = False

        return state

    # ======================================================
    # Supervisor Node (LLM Router)
    # ======================================================

    def supervisor(self, state):
        agents_text = "\n\n".join(
            [
                f"{name}: {description}"
                for name, description in AGENT_REGISTRY.items()
            ]
        )

        prompt = f"""
        You are a Legal AI Supervisor.
        Your task is to select the SINGLE best agent for the user's request.
        Available Agents:
        {agents_text}
        User Question:
        {state["question"]}
        Rules:
        1. Choose exactly one agent.
        2. Return only the agent name.
        3. No explanation.
        4. Must be one of:

        {", ".join(AGENT_REGISTRY.keys())}
        """
        try:
            response = self.router_llm.invoke(prompt)
            route = response.agent.strip().lower()
            if route not in AGENT_REGISTRY:
                route = "qa"
        except Exception:
            route = "qa"
        state["route"] = route
        return state

    # ======================================================
    # Agent Nodes
    # ======================================================

    def qa_node(self, state):
        return self._run(
            self._build(QAAgent),
            state
        )

    def clause_node(self, state):
        return self._run(
            self._build(ClauseExtractionAgent),
            state
        )

    def risk_node(self, state):
        return self._run(
            self._build(RiskAnalysisAgent),
            state
        )

    def compliance_node(self, state):
        return self._run(
            self._build(ComplianceAgent),
            state
        )

    def comparison_node(self, state):
        return self._run(
            self._build(ContractComparisonAgent),
            state
        )

    def obligation_node(self, state):
        return self._run(
            self._build(ObligationAgent),
            state
        )

    def report_node(self, state):
        return self._run(
            self._build(ReportGeneratorAgent),
            state
        )

    # ======================================================
    # Routing Functions
    # ======================================================

    def after_guardrail(self, state):

        if state["blocked"]:
            return "end"

        return "supervisor"

    def after_supervisor(self, state):

        return state["route"]

    # ======================================================
    # Graph Builder
    # ======================================================

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
            {
                "end": END,
                "supervisor": "supervisor",
            },
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

    # ======================================================
    # Entry Function
    # ======================================================

    def route(
        self,
        question,
        has_document=False,
        general=False,
    ):

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