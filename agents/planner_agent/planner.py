"""
Planner (Supervisor) Agent  -  LangGraph, registry-driven, one node per agent
Architecture (proper LangGraph multi-agent supervisor):

        START
          v
      [guardrail]  --blocked--> END
          v (ok)
      [supervisor]  (classifies intent -> chooses an agent key)
          v (conditional edges)
   +----+----+----+----+----+----+----+
   v    v    v    v    v    v    v
  [qa][clause][risk][compliance][comparison][obligation][report]   <- each agent = a NODE
   +----+----+----+----+----+----+----+
          v
         END

Every agent is registered in AGENT_REGISTRY. Each registry entry becomes a
node in the graph automatically, so adding a new agent = one registry line.
Falls back to plain routing if langgraph is not installed.
"""

from typing import TypedDict, Optional

from guardrails.guardrails import validate_input, is_off_topic, add_disclaimer

# ---- the specialized agents (all LangChain) ----
from agents.legal_rag_agent.qa_agent import QAAgent
from agents.clause_extraction_agent.clause_extractor import ClauseExtractionAgent
from agents.risk_analysis_agent.risk_agent import RiskAnalysisAgent
from agents.compliance_agent.compliance_agent import ComplianceAgent
from agents.contract_comparison_agent.comparison_agent import ContractComparisonAgent


try:
    from langgraph.graph import StateGraph, START, END
    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False


# =========================================================================
# 1. AGENT REGISTRY  -  add an agent here and it automatically becomes a node
#    key: {class, keywords that route to it}
# =========================================================================
AGENT_REGISTRY = {
    "risk": {
        "class": RiskAnalysisAgent,
        "keywords": ["risk", "risky", "red flag", "danger"],
    },
    "compliance": {
        "class": ComplianceAgent,
        "keywords": ["compliance", "compliant", "regulation", "gdpr"],
    },
    "comparison": {
        "class": ContractComparisonAgent,
        "keywords": ["compare", "difference", "version", "vs "],
    },
    "clause": {
        "class": ClauseExtractionAgent,
        "keywords": ["clause", "extract", "termination", "confidential", "liability"],
    },
    "qa": {                      # default fallback agent
        "class": QAAgent,
        "keywords": [],
    },
}

DEFAULT_AGENT = "qa"


def classify_intent(question: str) -> str:
    """Return the registry key of the agent that should handle this question."""
    q = question.lower()
    for key, cfg in AGENT_REGISTRY.items():
        if any(kw in q for kw in cfg["keywords"]):
            return key
    return DEFAULT_AGENT


# =========================================================================
# 2. Shared graph state
# =========================================================================
class PlannerState(TypedDict):
    question: str
    has_document: bool
    general: bool
    route: Optional[str]
    agent: Optional[str]
    answer: Optional[str]
    mode: Optional[str]
    blocked: bool


# =========================================================================
# 3. Planner
# =========================================================================
class Planner:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.graph = self._build_graph() if HAS_LANGGRAPH else None

    # ---- build an agent instance from the registry, with shared components ----
    def _build_agent(self, key):
        self.pipeline._get_knowledge_retriever()
        agent_cls = AGENT_REGISTRY[key]["class"]
        return agent_cls(
            knowledge_retriever=self.pipeline.knowledge_retriever,
            document_retriever=self.pipeline.document_retriever,
            generator=self.pipeline.generator,
            memory=self.pipeline.memory,
            router=self.pipeline.router,
        )

    # ================= GRAPH NODES =================
    def _node_guardrail(self, state: PlannerState) -> PlannerState:
        ok, msg = validate_input(state["question"])
        if not ok:
            state.update(blocked=True, agent="Guardrail", answer=msg, mode="blocked")
            return state
        if is_off_topic(state["question"]):
            state.update(blocked=True, agent="Guardrail",
                         answer="I'm a legal assistant, so I can only help with legal "
                                "documents and legal questions.",
                         mode="blocked")
            return state
        state["blocked"] = False
        return state

    def _node_supervisor(self, state: PlannerState) -> PlannerState:
        # the supervisor decides which agent node to route to
        state["route"] = classify_intent(state["question"])
        return state

    def _make_agent_node(self, key):
        """Factory: build a graph node function for one registry agent."""
        def node(state: PlannerState) -> PlannerState:
            agent = self._build_agent(key)
            result = agent.run(state["question"], state["has_document"], state["general"])
            state["agent"] = result["agent"]
            state["answer"] = add_disclaimer(result["answer"])
            state["mode"] = result["mode"]
            return state
        return node

    # ================= EDGE DECISIONS =================
    def _after_guardrail(self, state: PlannerState) -> str:
        return "blocked" if state["blocked"] else "supervisor"

    def _after_supervisor(self, state: PlannerState) -> str:
        # return the agent node name to jump to
        return state["route"]

    # ================= BUILD THE GRAPH =================
    def _build_graph(self):
        g = StateGraph(PlannerState)

        # core nodes
        g.add_node("guardrail", self._node_guardrail)
        g.add_node("supervisor", self._node_supervisor)

        # ONE NODE PER AGENT (from the registry)
        for key in AGENT_REGISTRY:
            g.add_node(key, self._make_agent_node(key))

        # START -> guardrail
        g.add_edge(START, "guardrail")

        # guardrail -> (blocked ? END : supervisor)
        g.add_conditional_edges("guardrail", self._after_guardrail,
                                {"blocked": END, "supervisor": "supervisor"})

        # supervisor -> the chosen agent node (conditional edges to every agent)
        agent_paths = {key: key for key in AGENT_REGISTRY}
        g.add_conditional_edges("supervisor", self._after_supervisor, agent_paths)

        # every agent node -> END
        for key in AGENT_REGISTRY:
            g.add_edge(key, END)

        return g.compile()

    # ================= PUBLIC ENTRY =================
    def route(self, question, has_document=False, general=False):
        if has_document:
            self.pipeline._get_document_retriever()

        if HAS_LANGGRAPH and self.graph is not None:
            state: PlannerState = {
                "question": question, "has_document": has_document, "general": general,
                "route": None, "agent": None, "answer": None, "mode": None, "blocked": False,
            }
            out = self.graph.invoke(state)
            return {"agent": out["agent"], "answer": out["answer"], "mode": out["mode"]}

        # ---- fallback without langgraph ----
        ok, msg = validate_input(question)
        if not ok:
            return {"agent": "Guardrail", "answer": msg, "mode": "blocked"}
        if is_off_topic(question):
            return {"agent": "Guardrail",
                    "answer": "I'm a legal assistant and can only help with legal questions.",
                    "mode": "blocked"}
        key = classify_intent(question)
        agent = self._build_agent(key)
        result = agent.run(question, has_document, general)
        result["answer"] = add_disclaimer(result["answer"])
        return result
